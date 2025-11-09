from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(order=True)
class BeamNode:
    sequence: List[str]
    g_cost: float
    h_cost: float
    f_cost: float
    covered_tokens: Set[str]
    detail: Dict[str, float]


class AStarBeamGenerator:
    """N-gram-based beam search comment generator with extensible heuristics."""

    def __init__(
        self,
        *,
        beam_width: int = 3,
        alpha: float = 0.1,
        beta: float = 0.5,
        gamma: float = 0.2,
        target_length: int = 20,
        use_semantic_heuristic: bool = True,
        lambda_future: float = 0.2,
        future_window: int = 5,
        max_expansions: int = 20000,
    ) -> None:
        self.beam_width = beam_width
        self.target_length = target_length
        self.use_semantic_heuristic = use_semantic_heuristic
        self.future_window = future_window
        self.max_expansions = max_expansions

        self.base_weights = {
            "length": alpha,
            "coverage": beta,
            "semantic": gamma,
            "future": lambda_future,
        }

        self.n_gram = 3
        self.transitions: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
        self.context_counts: Counter = Counter()
        self.vocab_probs: Counter = Counter()
        self.token_doc_freq: Counter = Counter()
        self.discriminative_scores: Dict[str, float] = {}

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.code_embedding: Optional[np.ndarray] = None

        self.code_tokens: Set[str] = set()
        self.code_token_list: List[str] = []
        self.identifier_windows: Dict[str, Tuple[int, int]] = {}
        self.identifier_weights: Dict[str, float] = {}

        self._latest_components: Dict[str, float] = {}
        self.last_trace: List[Dict[str, object]] = []

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        comments: Iterable[str],
        *,
        code_data: Optional[Iterable[str]] = None,
        n_gram: int = 3,
    ) -> None:
        comments = [c for c in comments if c]
        if not comments:
            raise ValueError("No training comments provided.")

        self.n_gram = max(2, n_gram)
        self.transitions = defaultdict(dict)
        self.context_counts = Counter()
        self.vocab_probs = Counter()
        self.token_doc_freq = Counter()

        for comment in comments:
            tokens = self._tokenize(comment)
            unique_tokens = set(tokens)
            for tok in unique_tokens:
                self.token_doc_freq[tok] += 1

            tokens = ["<START>"] * (self.n_gram - 1) + tokens + ["<END>"]
            for i in range(len(tokens) - self.n_gram + 1):
                context = tuple(tokens[i : i + self.n_gram - 1])
                next_token = tokens[i + self.n_gram - 1]
                self.context_counts[context] += 1
                self.transitions.setdefault(context, {})
                self.transitions[context][next_token] = (
                    self.transitions[context].get(next_token, 0) + 1
                )
                self.vocab_probs[next_token] += 1

        for context, counts in self.transitions.items():
            total = float(self.context_counts[context] + len(counts))
            for token in list(counts.keys()):
                counts[token] = (counts[token] + 1.0) / total

        total_tokens = float(sum(self.vocab_probs.values()))
        if total_tokens:
            for token in list(self.vocab_probs.keys()):
                self.vocab_probs[token] /= total_tokens

        num_docs = float(max(len(comments), 1))
        self.discriminative_scores = {
            token: math.log((num_docs + 1.0) / (df + 1.0)) + 1.0
            for token, df in self.token_doc_freq.items()
        }

        code_corpus = list(code_data) if code_data else []
        if self.use_semantic_heuristic and code_corpus:
            corpus = code_corpus + comments
            self.vectorizer = TfidfVectorizer(max_features=500, lowercase=True)
            self.vectorizer.fit(corpus)
        else:
            self.vectorizer = None
            self.use_semantic_heuristic = False

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        code_tokens: List[str],
        code_text: str = "",
        *,
        max_length: int = 32,
        return_metadata: bool = False,
    ):
        if not self.transitions:
            raise RuntimeError("Generator has not been trained or loaded.")

        self.code_token_list = [tok.lower() for tok in code_tokens if tok]
        self.code_tokens = set(self.code_token_list)
        self.identifier_windows = self._build_identifier_windows(max_length)
        self.identifier_weights = self._build_identifier_weights()

        self.code_embedding = None
        if self.use_semantic_heuristic and self.vectorizer and code_text:
            self.code_embedding = self.vectorizer.transform(
                [code_text.lower()]
            ).toarray()[0]

        start_sequence = ["<START>"] * (self.n_gram - 1)
        start_h = self._heuristic(start_sequence, set())
        beam = [
            BeamNode(
                sequence=start_sequence,
                g_cost=0.0,
                h_cost=start_h,
                f_cost=start_h,
                covered_tokens=set(),
                detail=self._latest_components.copy(),
            )
        ]
        completed: List[BeamNode] = []
        expansions = 0

        stop = False
        for _ in range(max_length):
            candidates: List[BeamNode] = []
            for node in beam:
                if node.sequence and node.sequence[-1] == "<END>":
                    completed.append(node)
                    continue

                context = tuple(node.sequence[-(self.n_gram - 1) :])
                next_options = self.transitions.get(context)
                if not next_options:
                    completed.append(node)
                    continue

                for token, prob in next_options.items():
                    new_sequence = node.sequence + [token]
                    g_cost = node.g_cost + (-math.log(prob + 1e-12))
                    if g_cost > node.g_cost + 10.0:
                        continue

                    covered = set(node.covered_tokens)
                    if token.lower() in self.code_tokens:
                        covered.add(token.lower())

                    h_cost = self._heuristic(new_sequence, covered)
                    detail_snapshot = self._latest_components.copy()
                    candidates.append(
                        BeamNode(
                            sequence=new_sequence,
                            g_cost=g_cost,
                            h_cost=h_cost,
                            f_cost=g_cost + h_cost,
                            covered_tokens=covered,
                            detail=detail_snapshot,
                        )
                    )
                    expansions += 1
                    if expansions >= self.max_expansions:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                if candidates:
                    candidates.sort(key=lambda node: node.f_cost)
                    beam = candidates[: self.beam_width]
                break

            if not candidates:
                break

            candidates.sort(key=lambda node: node.f_cost)
            beam = candidates[: self.beam_width]

        all_final = completed + beam
        if not all_final:
            comment = "Unable to generate comment."
            metadata = {
                "log_prob": float("-inf"),
                "avg_log_prob": float("-inf"),
                "token_count": 0,
                "terminated": False,
                "expansions": expansions,
            }
            return (comment, metadata) if return_metadata else comment

        best = min(all_final, key=lambda node: node.f_cost)
        comment_tokens = [
            tok for tok in best.sequence if tok not in {"<START>", "<END>"}
        ]
        comment = " ".join(comment_tokens)
        terminated = bool(best.sequence and best.sequence[-1] == "<END>")

        log_prob = -best.g_cost
        token_count = len(comment_tokens)
        avg_log_prob = (
            log_prob / token_count if token_count > 0 else float("-inf")
        )

        metadata = {
            "log_prob": log_prob,
            "avg_log_prob": avg_log_prob,
            "token_count": token_count,
            "terminated": terminated,
            "expansions": expansions,
            "detail": best.detail,
        }
        if return_metadata:
            return comment, metadata
        return comment

    # ------------------------------------------------------------------ #
    # Saving / Loading
    # ------------------------------------------------------------------ #

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        config = {
            "beam_width": self.beam_width,
            "target_length": self.target_length,
            "use_semantic_heuristic": self.use_semantic_heuristic,
            "future_window": self.future_window,
            "base_weights": self.base_weights,
            "n_gram": self.n_gram,
            "max_expansions": self.max_expansions,
        }

        transitions_json = {
            repr(ctx): probs for ctx, probs in self.transitions.items()
        }
        context_counts_json = {repr(ctx): count for ctx, count in self.context_counts.items()}

        (directory / "config.json").write_text(json.dumps(config, indent=2))
        (directory / "transitions.json").write_text(json.dumps(transitions_json))
        (directory / "context_counts.json").write_text(json.dumps(context_counts_json))
        (directory / "vocab_probs.json").write_text(json.dumps(self.vocab_probs))
        (directory / "token_doc_freq.json").write_text(json.dumps(self.token_doc_freq))
        (directory / "discriminative_scores.json").write_text(
            json.dumps(self.discriminative_scores)
        )
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, directory / "vectorizer.joblib")

    @classmethod
    def load(cls, directory: Path) -> "AStarBeamGenerator":
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        generator = cls(
            beam_width=config["beam_width"],
            alpha=config["base_weights"]["length"],
            beta=config["base_weights"]["coverage"],
            gamma=config["base_weights"]["semantic"],
            target_length=config["target_length"],
            use_semantic_heuristic=config["use_semantic_heuristic"],
            lambda_future=config["base_weights"].get("future", 0.0),
            future_window=config["future_window"],
            max_expansions=config.get("max_expansions", 20000),
        )
        generator.base_weights = config["base_weights"]
        generator.n_gram = config["n_gram"]

        transitions_json = json.loads((directory / "transitions.json").read_text())
        context_counts_json = json.loads((directory / "context_counts.json").read_text())
        generator.transitions = defaultdict(
            dict,
            {tuple(ast.literal_eval(ctx)): probs for ctx, probs in transitions_json.items()},
        )
        generator.context_counts = Counter(
            {tuple(ast.literal_eval(ctx)): count for ctx, count in context_counts_json.items()}
        )
        generator.vocab_probs = Counter(
            json.loads((directory / "vocab_probs.json").read_text())
        )
        generator.token_doc_freq = Counter(
            json.loads((directory / "token_doc_freq.json").read_text())
        )
        generator.discriminative_scores = json.loads(
            (directory / "discriminative_scores.json").read_text()
        )

        vectorizer_path = directory / "vectorizer.joblib"
        if vectorizer_path.exists():
            generator.vectorizer = joblib.load(vectorizer_path)
        else:
            generator.vectorizer = None

        return generator

    # ------------------------------------------------------------------ #
    # Heuristics
    # ------------------------------------------------------------------ #

    def _heuristic(self, sequence: List[str], covered_tokens: Set[str]) -> float:
        actual_tokens = [t for t in sequence if t not in {"<START>", "<END>"}]
        step = len(actual_tokens)

        components = {
            "length": self._length_bias(step),
            "coverage": self._coverage_penalty(covered_tokens),
            "semantic": self._semantic_distance(actual_tokens),
            "future": self._future_regret(step, covered_tokens),
        }
        total = 0.0
        for key, value in components.items():
            total += self.base_weights.get(key, 0.0) * value
        self._latest_components = components
        return total

    def _length_bias(self, step: int) -> float:
        return abs(step - self.target_length) * 0.2

    def _coverage_penalty(self, covered_tokens: Set[str]) -> float:
        if not self.code_tokens:
            return 0.0
        total_weight = 0.0
        uncovered_weight = 0.0
        for token in self.code_tokens:
            weight = self.identifier_weights.get(token, 1.0)
            total_weight += weight
            if token not in covered_tokens:
                uncovered_weight += weight
        if total_weight == 0:
            return 0.0
        return uncovered_weight / total_weight

    def _semantic_distance(self, tokens: List[str]) -> float:
        if not tokens or not self.use_semantic_heuristic or self.vectorizer is None:
            return 0.0
        if self.code_embedding is None or np.linalg.norm(self.code_embedding) == 0:
            return 0.0

        seq_embedding = self.vectorizer.transform([" ".join(tokens)]).toarray()[0]
        norm_product = np.linalg.norm(self.code_embedding) * np.linalg.norm(seq_embedding)
        if norm_product == 0:
            return 0.0
        cosine_sim = float(np.dot(self.code_embedding, seq_embedding) / norm_product)
        return max(0.0, 1.0 - cosine_sim)

    def _future_regret(self, step: int, covered_tokens: Set[str]) -> float:
        if not self.identifier_windows:
            return 0.0
        penalty = 0.0
        for token, (start, end) in self.identifier_windows.items():
            if token in covered_tokens:
                continue
            if step < start:
                urgency = 1.0
            elif step <= end:
                window_span = max(end - start + 1, 1)
                urgency = 1.0 - (step - start) / window_span
            else:
                urgency = 1.5
            weight = self.identifier_weights.get(token, 1.0)
            penalty += weight * urgency
        return penalty

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_identifier_windows(self, max_length: int) -> Dict[str, Tuple[int, int]]:
        windows: Dict[str, Tuple[int, int]] = {}
        if not self.code_token_list:
            return windows
        for idx, token in enumerate(self.code_token_list):
            if token in windows:
                continue
            start = max(1, idx + 1)
            end = min(max_length, start + self.future_window)
            windows[token] = (start, end)
        return windows

    def _build_identifier_weights(self) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        if not self.code_token_list:
            return weights
        for token in self.code_token_list:
            weights[token] = self.discriminative_scores.get(token, 1.0)
        if not weights:
            return weights
        max_weight = max(weights.values())
        if max_weight == 0:
            return weights
        for token in list(weights.keys()):
            weights[token] = weights[token] / max_weight
        return weights

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [tok for tok in text.split() if tok]


