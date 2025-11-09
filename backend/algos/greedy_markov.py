"""
Greedy Markov Comment Generator (advanced variants G0-G4).

Implements the progression of greedy baselines described in the project plan:

- G0: base n-gram greedy decoding
- G1: + cache / recency boost
- G2: + semantic anchors / identifier urgency
- G3: + one-step look-ahead scoring
- G4: + bounded backtracking when confidence is low

Additional utilities:
- Optional nucleus sampling + coverage bonus for stochastic trials
- Detailed per-step logging for analysis/ablation studies
"""

import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

class GreedyMarkovGenerator:
    def __init__(
        self,
        n_gram: int = 4,
        backoff_weight: float = 0.5,
        entropy_threshold: float = 2.0,
        lambda_anchor: float = 0.0,
        lambda_cache: float = 0.0,
        lookahead_k: int = 0,
        backtrack_budget: int = 0,
        backtrack_span: int = 3,
        backtrack_gap_threshold: float = 0.0,
        nucleus_p: Optional[float] = None,
        coverage_bonus: float = 0.0,
        log_details: bool = False,
    ):
        """
        Args:
            n_gram: Maximum n-gram order (2-4 recommended).
            backoff_weight: Weight multiplier applied when backing off to lower n.
            entropy_threshold: Threshold for switching to higher-order contexts.
            lambda_anchor: Semantic anchor boosting weight (activates G2+ when >0).
            lambda_cache: Cache/recency bonus weight (activates G1+ when >0).
            lookahead_k: Number of top candidates to evaluate with one-step look-ahead (G3).
            backtrack_budget: Number of backtracking interventions allowed per decoding (G4).
            backtrack_span: How many tokens to rewind when backtracking triggers.
            backtrack_gap_threshold: Confidence gap threshold to trigger backtracking.
            nucleus_p: Optional nucleus sampling cumulative probability (0<p<=1) for stochastic runs.
            coverage_bonus: Additional boost if a candidate mentions uncovered anchors (used with nucleus sampling).
            log_details: When True, store per-step decision metadata in `self.last_run_trace`.
        """

        if n_gram < 2:
            raise ValueError("n_gram must be >= 2")

        self.n_gram = n_gram
        self.backoff_weight = backoff_weight
        self.entropy_threshold = entropy_threshold
        self.lambda_anchor = lambda_anchor
        self.lambda_cache = lambda_cache
        self.lookahead_k = lookahead_k
        self.backtrack_budget = backtrack_budget
        self.backtrack_span = backtrack_span
        self.backtrack_gap_threshold = backtrack_gap_threshold
        self.nucleus_p = nucleus_p
        self.coverage_bonus = coverage_bonus
        self.log_details = log_details

        # Transition tables: {(context,): {next_word: prob}}
        self.transitions: Dict[int, Dict[Tuple[str, ...], Dict[str, float]]] = {
            i: defaultdict(Counter) for i in range(1, n_gram + 1)
        }
        self.context_counts: Dict[int, Counter] = {
            i: Counter() for i in range(1, n_gram + 1)
        }

        # Runtime state (populated during generate)
        self.anchors: set = set()
        self.anchor_weights: Dict[str, float] = {}
        self.code_complexity: float = 0.5
        self.last_run_trace: List[Dict[str, object]] = []
        self._cache_counter: Counter = Counter()
        self._history_tokens: List[str] = []
        
    def train(self, comments: List[str]):
        """Build n-gram transition tables from comment corpus."""
        print(f"Training on {len(comments)} comments...")
        
        for comment in comments:
            tokens = self._tokenize(comment)
            tokens = ['<START>'] * (self.n_gram - 1) + tokens + ['<END>']
            
            # Build transitions for all n-gram orders
            for i in range(len(tokens)):
                for n in range(1, self.n_gram + 1):
                    if i >= n:
                        context = tuple(tokens[i-n:i])
                        next_token = tokens[i]
                        self.transitions[n][context][next_token] += 1
                        self.context_counts[n][context] += 1
        
        # Convert counts to probabilities
        for n in range(1, self.n_gram + 1):
            for context in self.transitions[n]:
                total = self.context_counts[n][context]
                for word in self.transitions[n][context]:
                    self.transitions[n][context][word] /= total
        
        print("Training complete!")
    
    def generate(self, code_tokens: List[str], max_length: int = 20) -> str:
        """
        Generate comment for given code using greedy decoding with adaptive n-grams.
        
        Args:
            code_tokens: Important tokens from code (function names, params, etc.)
            max_length: Maximum comment length
        
        Returns:
            Generated comment string
        """
        # Set semantic anchors from code tokens with expanded vocabulary
        self.anchors, self.anchor_weights = self._expand_anchors_with_weights(code_tokens)
        self._cache_counter = Counter()
        self.last_run_trace = []
        self._history_tokens = []

        # Estimate code complexity (simple heuristic: based on number of tokens)
        self.code_complexity = min(len(code_tokens) / 10.0, 1.0)

        # Initialize with START tokens
        sequence: List[str] = ['<START>'] * (self.n_gram - 1)
        chosen_tokens: List[str] = []
        self._history_tokens = chosen_tokens

        backtrack_budget = self.backtrack_budget

        for step in range(max_length):
            distribution, analysis = self._candidate_distribution(sequence, chosen_tokens)

            if not distribution:
                break

            ranked = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            top_token, top_score = ranked[0]
            gap = (top_score - ranked[1][1]) if len(ranked) > 1 else float('inf')

            # Optional nucleus sampling for stochastic experiments
            if self.nucleus_p:
                top_token = self._sample_from_nucleus(distribution, chosen_tokens)

            sequence.append(top_token)
            if top_token != '<END>':
                chosen_tokens.append(top_token)
            self._cache_counter[top_token] += 1

            # Logging for analysis/debugging
            if self.log_details:
                trace_entry = {
                    'step': step,
                    'context': tuple(sequence[-self.n_gram:]),
                    'top_token': top_token,
                    'score': top_score,
                    'gap': gap,
                    'distribution': ranked[: min(10, len(ranked))],
                }
                trace_entry.update(analysis)
                self.last_run_trace.append(trace_entry)

            if top_token == '<END>':
                break

            # Backtracking: trigger when gap is small (low confidence)
            if (
                backtrack_budget > 0
                and self.backtrack_gap_threshold > 0
                and gap < self.backtrack_gap_threshold
            ):
                rewind = min(self.backtrack_span, len(chosen_tokens))
                if rewind > 0:
                    del sequence[-rewind:]
                    del chosen_tokens[-rewind:]
                    backtrack_budget -= 1
                    continue

        # Remove START tokens and join
        comment_tokens = [
            t for t in sequence[self.n_gram - 1 :]
            if t not in ('<END>', '<START>')
        ]
        comment = ' '.join(comment_tokens).strip()
        return comment

    # ------------------------------------------------------------------
    # Candidate generation & scoring utilities
    # ------------------------------------------------------------------

    def _candidate_distribution(
        self, sequence: List[str], chosen_tokens: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, object]]:
        """Compute candidate scores with optional heuristics."""

        optimal_n = self._select_n_gram_order(sequence)
        base_dist: Dict[str, float] = defaultdict(float)
        analysis: Dict[str, object] = {
            'selected_n': optimal_n,
            'entropy': None,
            'cache_scores': {},
            'anchor_scores': {},
            'lookahead_scores': {},
        }

        # Aggregate probabilities with geometric backoff
        for n in range(optimal_n, 0, -1):
            context = tuple(sequence[-n:])
            if context not in self.transitions[n]:
                continue
            weight = self.backoff_weight ** (optimal_n - n)
            for word, prob in self.transitions[n][context].items():
                base_dist[word] += weight * prob

            if analysis['entropy'] is None:
                analysis['entropy'] = self._calculate_entropy(
                    self.transitions[n][context]
                )

        if not base_dist:
            return {}, analysis

        # Normalize base distribution
        total = sum(base_dist.values())
        for word in list(base_dist.keys()):
            base_dist[word] = base_dist[word] / total

        # Apply heuristic adjustments in log-space
        scores: Dict[str, float] = {}
        for word, prob in base_dist.items():
            log_p = math.log(prob + 1e-12)

            if self.lambda_cache > 0:
                cache_val = self._cache_score(word, len(chosen_tokens))
                log_p += self.lambda_cache * cache_val
                analysis['cache_scores'][word] = cache_val

            if self.lambda_anchor > 0:
                anchor_val = self._anchor_boost(word, len(chosen_tokens))
                log_p += self.lambda_anchor * anchor_val
                analysis['anchor_scores'][word] = anchor_val

            scores[word] = log_p

        # One-step look-ahead (only affects ranking, not probabilities)
        if self.lookahead_k > 0:
            top_candidates = sorted(scores, key=scores.get, reverse=True)[: self.lookahead_k]
            for word in top_candidates:
                la_score = self._look_ahead(sequence, word)
                scores[word] += la_score
                analysis['lookahead_scores'][word] = la_score

        # Convert back to positive scores for argmax / sampling
        max_log = max(scores.values())
        distribution = {
            word: math.exp(score - max_log) for word, score in scores.items()
        }

        # Optional coverage bonus (encourages unseen anchors)
        if self.coverage_bonus > 0:
            uncovered = self.anchors - set(t.lower() for t in chosen_tokens)
            for word in distribution:
                if word.lower() in uncovered:
                    distribution[word] *= (1 + self.coverage_bonus)

        # Final normalization
        total = sum(distribution.values())
        for word in list(distribution.keys()):
            distribution[word] = distribution[word] / total

        return distribution, analysis
    
    def _select_n_gram_order(self, sequence: List[str]) -> int:
        """
        Dynamically select n-gram order based on:
        1. Context entropy (higher entropy -> higher n needed)
        2. Code complexity (higher complexity -> higher n preferred)
        """
        # Start with base n-gram order
        base_n = 2
        
        # Check entropy of current context
        best_n = base_n

        for n in range(2, min(self.n_gram + 1, len(sequence) + 1)):
            if len(sequence) < n:
                continue

            context = tuple(sequence[-(n):])
            
            # Calculate entropy for this context
            if context in self.transitions[n]:
                entropy = self._calculate_entropy(self.transitions[n][context])
                
                # If entropy is high, we need more context (higher n)
                if entropy > self.entropy_threshold:
                    best_n = n
        
        # Also consider code complexity in selection
        complexity_bonus = int(self.code_complexity * 2)  # 0, 1, or 2 extra
        selected_n = min(best_n + complexity_bonus, self.n_gram)
        
        # Ensure minimum of 2 for some context
        return max(2, selected_n)
    
    def _calculate_entropy(self, prob_dist: dict) -> float:
        """Calculate Shannon entropy of a probability distribution."""
        entropy = 0.0
        for prob in prob_dist.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy
    
    def _cache_score(self, word: str, position: int) -> float:
        """Recency-weighted cache score."""
        if not self._history_tokens:
            return 0.0

        score = 0.0
        for idx, token in enumerate(reversed(self._history_tokens)):
            if token == word:
                score += 1.0 / (idx + 1)
        return score

    def _anchor_boost(self, word: str, position: int) -> float:
        """Compute urgency-weighted anchor boost."""
        word_lower = word.lower()
        if word_lower not in self.anchors:
            return 0.0
        urgency = self.anchor_weights.get(word_lower, 1.0)
        decay = 1.0 / (1 + position)
        return urgency * decay

    def _look_ahead(self, sequence: List[str], candidate: str) -> float:
        """Approximate future gain by peeking one token ahead."""
        if candidate == '<END>':
            return 0.0

        temp_sequence = sequence + [candidate]
        optimal_n = self._select_n_gram_order(temp_sequence)

        best_future = 0.0
        for n in range(optimal_n, 0, -1):
            context = tuple(temp_sequence[-n:])
            if context not in self.transitions[n]:
                continue
            best_future = max(best_future, max(self.transitions[n][context].values()))
            break

        return math.log(best_future + 1e-12)

    def _sample_from_nucleus(
        self, distribution: Dict[str, float], chosen_tokens: List[str]
    ) -> str:
        threshold = min(max(self.nucleus_p, 1e-6), 1.0)
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        cumulative = 0.0
        nucleus = []
        for token, prob in sorted_items:
            nucleus.append((token, prob))
            cumulative += prob
            if cumulative >= threshold:
                break

        rand = random.random() * cumulative
        running = 0.0
        for token, prob in nucleus:
            running += prob
            if rand <= running:
                return token
        return nucleus[-1][0]

    def _expand_anchors_with_weights(self, code_tokens: List[str]) -> Tuple[set, Dict[str, float]]:
        """
        Expand semantic anchors and provide urgency weights for identifier coverage.
        """
        anchors: set = set()
        weights: Dict[str, float] = {}
        
        # Simple lexical mapping
        token_mappings = {
            'add': ['add', 'sum', 'plus'],
            'calculate': ['calculate', 'compute', 'determine'],
            'return': ['return', 'give', 'provide'],
            'get': ['get', 'fetch', 'retrieve'],
            'set': ['set', 'assign', 'update'],
            'find': ['find', 'search', 'locate'],
            'check': ['check', 'verify', 'validate'],
            'create': ['create', 'make', 'generate'],
            'delete': ['delete', 'remove', 'eliminate'],
            'update': ['update', 'modify', 'change'],
            'parse': ['parse', 'analyze', 'read'],
            'format': ['format', 'convert', 'transform'],
        }
        
        if not code_tokens:
            return anchors, weights

        window = max(3, len(code_tokens))

        for idx, token in enumerate(code_tokens):
            token_lower = token.lower()
            anchors.add(token_lower)
            urgency = 1 - (idx / window)
            weights[token_lower] = max(0.1, urgency)
            
            # Add mapped words
            if token_lower in token_mappings:
                for mapped in token_mappings[token_lower]:
                    anchors.add(mapped)
                    weights.setdefault(mapped, urgency * 0.8)
        
        return anchors, weights
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be improved with nltk)."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def save(self, path: str) -> None:
        """
        Save trained model to disk (JSON format, no pickle).
        
        Args:
            path: Directory path where model files will be saved
        """
        import json
        from pathlib import Path
        
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert transitions to JSON-serializable format
        transitions_json = {}
        for n, trans_dict in self.transitions.items():
            transitions_json[str(n)] = {
                str(ctx): {w: float(p) for w, p in probs.items()}
                for ctx, probs in trans_dict.items()
            }
        
        context_counts_json = {
            str(n): {str(ctx): int(count) for ctx, count in counts.items()}
            for n, counts in self.context_counts.items()
        }
        
        config = {
            'n_gram': self.n_gram,
            'backoff_weight': self.backoff_weight,
            'entropy_threshold': self.entropy_threshold,
            'lambda_anchor': self.lambda_anchor,
            'lambda_cache': self.lambda_cache,
            'lookahead_k': self.lookahead_k,
            'backtrack_budget': self.backtrack_budget,
            'backtrack_span': self.backtrack_span,
            'backtrack_gap_threshold': self.backtrack_gap_threshold,
            'nucleus_p': self.nucleus_p,
            'coverage_bonus': self.coverage_bonus,
        }
        
        (model_dir / 'config.json').write_text(json.dumps(config, indent=2))
        (model_dir / 'transitions.json').write_text(json.dumps(transitions_json, indent=2))
        (model_dir / 'context_counts.json').write_text(json.dumps(context_counts_json, indent=2))
        
        print(f"Model saved to {model_dir}")
    
    @classmethod
    def load(cls, path: str) -> 'GreedyMarkovGenerator':
        """
        Load trained model from disk.
        
        Args:
            path: Directory path containing model files
        
        Returns:
            Loaded GreedyMarkovGenerator instance
        """
        import json
        from pathlib import Path
        
        model_dir = Path(path)
        
        config = json.loads((model_dir / 'config.json').read_text())
        transitions_json = json.loads((model_dir / 'transitions.json').read_text())
        context_counts_json = json.loads((model_dir / 'context_counts.json').read_text())
        
        # Reconstruct generator
        gen = cls(**config)
        
        # Reconstruct transitions (ctx is already a tuple string, parse it)
        import ast
        gen.transitions = {
            int(n): defaultdict(Counter, {
                ast.literal_eval(ctx): Counter(probs)
                for ctx, probs in trans_dict.items()
            })
            for n, trans_dict in transitions_json.items()
        }
        
        gen.context_counts = {
            int(n): Counter({
                ast.literal_eval(ctx): count
                for ctx, count in counts.items()
            })
            for n, counts in context_counts_json.items()
        }
        
        print(f"Model loaded from {model_dir}")
        return gen


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    # Sample training data
    training_comments = [
        "calculate the sum of two numbers",
        "return the sum of a and b",
        "add two integers and return result",
        "compute sum of input parameters",
        "function to add numbers together"
    ]
    
    # Train model with adaptive n-grams (2-4)
    gen = GreedyMarkovGenerator(
        n_gram=4,
        entropy_threshold=2.0,
        lambda_cache=0.5,
        lambda_anchor=0.6,
        lookahead_k=5,
        backtrack_budget=1,
        backtrack_gap_threshold=0.3,
        log_details=True,
    )
    gen.train(training_comments)
    
    # Generate comment for code with semantic anchors
    code_tokens = ["sum", "numbers", "add"]
    comment = gen.generate(code_tokens, max_length=15)
    
    print(f"Generated: {comment}")
    print(f"\nModel features:")
    print(f"- Adaptive n-grams (entropy-based selection)")
    print(f"- Semantic anchor weighting (code tokens: {code_tokens})")
    print(f"- Code complexity: {gen.code_complexity:.2f}")