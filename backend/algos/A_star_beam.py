"""
A* Beam Search Comment Generator (advanced variants A0-A5).

Supports modular heuristics and staged weighting:
- Length bias, identifier coverage, semantic distance (baseline)
- Future-regret urgency, contrastive specificity, dependency ordering, probability risk
- Optional staged weighting schedules for different comment segments
"""

import heapq
import math
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Optional: For word embeddings (uncomment if you want to use embeddings)
# from gensim.models import Word2Vec
# or use spaCy: import spacy

class BeamNode:
    """Node representing a partial comment sequence."""

    def __init__(
        self,
        sequence: List[str],
        g_cost: float,
        h_cost: float,
        covered_tokens: Set[str],
        detail: Optional[Dict[str, float]] = None,
    ):
        self.sequence = sequence
        self.g_cost = g_cost  # Actual cost (neg log prob)
        self.h_cost = h_cost  # Allows tracking heuristic components
        self.f_cost = g_cost + h_cost
        self.covered_tokens = covered_tokens
        self.detail = detail or {}
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class AStarBeamGenerator:
    def __init__(
        self,
        beam_width: int = 5,
        alpha: float = 0.2,
        beta: float = 0.4,
        gamma: float = 0.4,
        target_length: int = 10,
        use_semantic_heuristic: bool = True,
        lambda_future: float = 0.0,
        lambda_contrastive: float = 0.0,
        lambda_dependency: float = 0.0,
        lambda_risk: float = 0.0,
        future_window: int = 4,
        future_eta: float = 0.6,
        future_kappa: float = 0.4,
        retrieval_k: int = 5,
        risk_alpha: float = 0.0,
        risk_beta: float = 0.0,
        stage_schedule: Optional[List[Dict]] = None,
        log_details: bool = False,
    ):
        """
        Args:
            beam_width: Number of candidates to maintain.
            alpha/beta/gamma: Base heuristic weights (length, coverage, semantic).
            lambda_future: Weight for future-regret urgency heuristic.
            lambda_contrastive: Weight for contrastive specificity reward.
            lambda_dependency: Weight for dependency ordering penalty.
            lambda_risk: Weight for probability-risk control.
            future_window: Preferred mention window size for identifiers.
            future_eta/future_kappa: Parameters for late penalty escalation.
            retrieval_k: Number of neighbor comments to approximate discriminative stats.
            risk_alpha/risk_beta: Coefficients for entropy and peak-risk terms.
            stage_schedule: Optional staged weighting (list of dicts with 'until' & 'weights').
            log_details: When True, stores per-node heuristic breakdowns.
        """

        self.beam_width = beam_width
        self.target_length = target_length
        self.use_semantic_heuristic = use_semantic_heuristic
        self.future_window = future_window
        self.future_eta = future_eta
        self.future_kappa = future_kappa
        self.retrieval_k = retrieval_k
        self.risk_alpha = risk_alpha
        self.risk_beta = risk_beta
        self.log_details = log_details

        self.base_weights = {
            'length': alpha,
            'coverage': beta,
            'semantic': gamma,
            'future': lambda_future,
            'contrastive': lambda_contrastive,
            'dependency': lambda_dependency,
            'risk': lambda_risk,
        }
        self.stage_schedule = stage_schedule or []

        # Transition probabilities (trained from corpus)
        self.transitions: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(Counter)
        self.context_counts: Counter = Counter()
        self.n_gram: int = 3

        # Training stats for heuristics
        self.vocab_probs: Counter = Counter()
        self.token_doc_freq: Counter = Counter()
        self.discriminative_scores: Dict[str, float] = {}
        self.training_comments: List[str] = []

        # Runtime state for generation
        self.code_tokens: Set[str] = set()
        self.code_token_list: List[str] = []
        self.code_text: str = ""
        self.identifier_windows: Dict[str, Tuple[int, int]] = {}
        self.identifier_weights: Dict[str, float] = {}
        self.dependency_pairs: List[Tuple[str, str, float]] = []
        self.last_trace: List[Dict[str, object]] = []

        # TF-IDF vectorizer for semantic distance
        self.vectorizer = None
        self.code_embedding = None
        self._latest_components: Dict[str, float] = {}
        
    def train(
        self,
        comments: List[str],
        code_data: Optional[List[str]] = None,
        n_gram: int = 3,
    ):
        """
        Build transition probability model.
        
        Args:
            comments: List of training comments
            code_data: List of corresponding code snippets (for TF-IDF training)
            n_gram: N-gram order
        """
        print(f"Training A* model on {len(comments)} comments...")
        self.n_gram = max(2, n_gram)
        self.transitions = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab_probs = Counter()
        self.token_doc_freq = Counter()
        self.training_comments = comments

        # Stage schedule sanity: ensure sorted by 'until'
        self.stage_schedule.sort(key=lambda cfg: cfg.get('until', float('inf')))

        for comment in comments:
            tokens = self._tokenize(comment)
            unique_tokens = set(tokens)
            for tok in unique_tokens:
                self.token_doc_freq[tok] += 1

            tokens = ['<START>'] * (self.n_gram - 1) + tokens + ['<END>']
            
            # Build n-gram transitions
            for i in range(len(tokens) - self.n_gram + 1):
                context = tuple(tokens[i : i + self.n_gram - 1])
                next_token = tokens[i + self.n_gram - 1]
                
                self.transitions[context][next_token] += 1
                self.context_counts[context] += 1
                self.vocab_probs[next_token] += 1
        
        # Convert to probabilities
        for context in self.transitions:
            total = self.context_counts[context]
            for word in self.transitions[context]:
                self.transitions[context][word] /= total
        
        # Global vocab probabilities
        total_tokens = sum(self.vocab_probs.values())
        for word in self.vocab_probs:
            self.vocab_probs[word] /= total_tokens
        
        # Discriminative IDF scores
        num_docs = max(len(comments), 1)
        self.discriminative_scores = {}
        for token, df in self.token_doc_freq.items():
            self.discriminative_scores[token] = math.log((num_docs + 1) / (df + 1)) + 1.0
        
        # Train TF-IDF vectorizer on code data if provided
        if code_data and self.use_semantic_heuristic and len(code_data) > 0:
            print("Training TF-IDF vectorizer for semantic heuristic...")
            self.vectorizer = TfidfVectorizer(max_features=500, lowercase=True)
            
            # Use both code and comments for better vocabulary
            all_texts = code_data + comments
            self.vectorizer.fit(all_texts)
        else:
            self.use_semantic_heuristic = False
            print("Semantic heuristic disabled (no code data provided)")
        
        print("Training complete!")
    
    def generate(
        self,
        code_tokens: List[str],
        code_text: str = "",
        max_length: int = 20,
    ) -> str:
        """
        Generate comment using A* beam search.
        
        Args:
            code_tokens: Important tokens from code (AST-extracted)
            code_text: Full code text for semantic similarity
            max_length: Maximum sequence length
        
        Returns:
            Best generated comment
        """
        self.code_token_list = [tok.lower() for tok in code_tokens]
        self.code_tokens = set(self.code_token_list)
        self.code_text = code_text.lower() if code_text else ""

        # Compute code embedding for semantic similarity
        if self.use_semantic_heuristic and self.vectorizer and self.code_text:
            self.code_embedding = self.vectorizer.transform([self.code_text]).toarray()[0]
        else:
            self.code_embedding = None

        self.identifier_windows = self._build_identifier_windows()
        self.identifier_weights = self._build_identifier_weights()
        self.dependency_pairs = self._build_dependency_pairs()
        self.last_trace = []

        # Initialize beam with START nodes according to n-gram
        start_sequence = ['<START>'] * (self.n_gram - 1)
        start_h = self._heuristic(start_sequence, set())
        start_node = BeamNode(
            sequence=start_sequence,
            g_cost=0.0,
            h_cost=start_h,
            covered_tokens=set(),
            detail=self._latest_components.copy(),
        )

        beam = [start_node]
        completed = []
        
        for step in range(max_length):
            candidates = []
            
            # Expand each node in current beam
            for node in beam:
                if node.sequence[-1] == '<END>':
                    completed.append(node)
                    continue
                
                # Get possible next tokens
                context = tuple(node.sequence[-(self.n_gram - 1):])
                
                if context not in self.transitions:
                    # No transitions found - end sequence
                    completed.append(node)
                    continue
                
                for next_token, prob in self.transitions[context].items():
                    # Calculate new costs
                    new_sequence = node.sequence + [next_token]
                    new_g = node.g_cost + (-math.log(prob + 1e-10))
                    
                    new_covered = node.covered_tokens.copy()
                    if next_token.lower() in self.code_tokens:
                        new_covered.add(next_token.lower())
                    
                    new_h = self._heuristic(new_sequence, new_covered)
                    detail_snapshot = self._latest_components.copy()
                    
                    new_node = BeamNode(
                        new_sequence,
                        new_g,
                        new_h,
                        new_covered,
                        detail=detail_snapshot,
                    )
                    candidates.append(new_node)
            
            if not candidates:
                break
            
            # Keep top beam_width candidates
            beam = heapq.nsmallest(self.beam_width, candidates, key=lambda n: n.f_cost)
        
        # Select best completed sequence
        all_final = completed + beam
        if not all_final:
            return "error generating comment"
        
        best = min(all_final, key=lambda n: n.f_cost)

        if self.log_details:
            self.last_trace = [
                {
                    'sequence': best.sequence,
                    'detail': best.detail,
                    'g_cost': best.g_cost,
                    'h_cost': best.h_cost,
                    'f_cost': best.f_cost,
                }
            ]

        # Remove START/END tokens
        comment_tokens = [t for t in best.sequence if t not in ['<START>', '<END>']]
        return ' '.join(comment_tokens)
    
    def _heuristic(self, sequence: List[str], covered_tokens: Set[str]) -> float:
        """Compose heuristic value from enabled components."""

        actual_tokens = [t for t in sequence if t not in {'<START>', '<END>'}]
        step = len(actual_tokens)
        weights = self._stage_weights(step)

        components = {
            'length': self._length_bias(step),
            'coverage': self._coverage_penalty(covered_tokens),
            'semantic': self._semantic_distance(actual_tokens),
            'future': self._future_regret(step, covered_tokens),
            'contrastive': self._contrastive_reward(actual_tokens),
            'dependency': self._dependency_penalty(actual_tokens),
            'risk': self._risk_penalty(sequence),
        }

        total = 0.0
        for key, value in components.items():
            weight = weights.get(key, 0.0)
            total += weight * value

        self._latest_components = components
        return total

    # ------------------------------------------------------------------
    # Heuristic component helpers
    # ------------------------------------------------------------------

    def _stage_weights(self, step: int) -> Dict[str, float]:
        weights = self.base_weights.copy()
        for stage in self.stage_schedule:
            limit = stage.get('until', float('inf'))
            if step <= limit:
                stage_weights = stage.get('weights', {})
                for key, value in stage_weights.items():
                    weights[key] = value
                break
        return weights

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

    def _semantic_distance(self, actual_tokens: List[str]) -> float:
        if not actual_tokens or not self.use_semantic_heuristic or self.code_embedding is None:
            return 0.0

        seq_text = ' '.join(actual_tokens)
        if not seq_text:
            return 0.0

        seq_embedding = self.vectorizer.transform([seq_text]).toarray()[0]
        dot_product = np.dot(self.code_embedding, seq_embedding)
        norm_product = np.linalg.norm(self.code_embedding) * np.linalg.norm(seq_embedding)
        if norm_product > 0:
            cosine_sim = dot_product / norm_product
            return 1.0 - cosine_sim
        return 1.0

    def _future_regret(self, step: int, covered_tokens: Set[str]) -> float:
        if not self.identifier_windows:
            return 0.0

        penalty = 0.0
        for token, window in self.identifier_windows.items():
            if token in covered_tokens:
                continue

            start, end = window
            weight = self.identifier_weights.get(token, 1.0)

            if step < start:
                urgency = 1.0
            elif start <= step <= end:
                window_span = max(end - start + 1, 1)
                urgency = 1.0 - (step - start) / window_span
            else:
                urgency = self.future_eta * math.exp(self.future_kappa * (step - end))

            penalty += weight * urgency
        return penalty

    def _contrastive_reward(self, actual_tokens: List[str]) -> float:
        if not actual_tokens or not self.discriminative_scores:
            return 0.0

        counts = Counter()
        reward = 0.0
        for token in actual_tokens:
            token_lower = token.lower()
            if token_lower in {'<start>', '<end>'}:
                continue
            counts[token_lower] += 1
            disc = self.discriminative_scores.get(token_lower, 0.0)
            reward += disc / counts[token_lower]

        return -reward

    def _dependency_penalty(self, actual_tokens: List[str]) -> float:
        if not self.dependency_pairs:
            return 0.0

        positions: Dict[str, int] = {}
        for idx, token in enumerate(actual_tokens):
            token_lower = token.lower()
            positions.setdefault(token_lower, idx)

        penalty = 0.0
        norm = max(self.target_length, 1)
        for src, tgt, weight in self.dependency_pairs:
            pos_src = positions.get(src)
            pos_tgt = positions.get(tgt)

            if pos_tgt is None:
                continue

            if pos_src is None:
                penalty += weight * 0.5
                continue

            if pos_tgt <= pos_src:
                distance = max(1, pos_src - pos_tgt)
                penalty += weight * min(1.0, distance / norm)

        return penalty

    def _risk_penalty(self, sequence: List[str]) -> float:
        if (self.risk_alpha == 0 and self.risk_beta == 0) or self.n_gram < 2:
            return 0.0

        context = tuple(sequence[-(self.n_gram - 1) :])
        if context not in self.transitions:
            return 0.0

        probs = list(self.transitions[context].values())
        if not probs:
            return 0.0

        entropy = 0.0
        for prob in probs:
            if prob > 0:
                entropy -= prob * math.log(prob + 1e-12)

        top_probs = sorted(probs, reverse=True)
        peak_ratio = top_probs[0] / (top_probs[1] + 1e-12) if len(top_probs) > 1 else top_probs[0]

        return self.risk_alpha * entropy - self.risk_beta * math.log(peak_ratio + 1e-12)

    # ------------------------------------------------------------------
    # Runtime structure helpers
    # ------------------------------------------------------------------

    def _build_identifier_windows(self) -> Dict[str, Tuple[int, int]]:
        windows: Dict[str, Tuple[int, int]] = {}
        if not self.code_token_list:
            return windows

        for idx, token in enumerate(self.code_token_list):
            if token in windows:
                continue
            start = max(1, idx + 1)
            end = min(self.target_length, start + self.future_window)
            windows[token] = (start, end)
        return windows

    def _build_identifier_weights(self) -> Dict[str, float]:
        weights: Dict[str, float] = {}
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

    def _build_dependency_pairs(self) -> List[Tuple[str, str, float]]:
        pairs: List[Tuple[str, str, float]] = []
        tokens = self.code_token_list
        for i, src in enumerate(tokens):
            for j in range(i + 1, len(tokens)):
                tgt = tokens[j]
                weight = 1.0 / (j - i)
                pairs.append((src, tgt, weight))
        return pairs
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def save(self, path: str) -> None:
        """
        Save trained model to disk (JSON + joblib for vectorizer, no pickle for models).
        
        Args:
            path: Directory path where model files will be saved
        """
        import json
        import joblib
        from pathlib import Path
        
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert transitions to JSON-serializable format
        transitions_json = {
            str(ctx): {w: float(p) for w, p in probs.items()}
            for ctx, probs in self.transitions.items()
        }
        
        context_counts_json = {
            str(ctx): int(count) for ctx, count in self.context_counts.items()
        }
        
        vocab_probs_json = {w: float(p) for w, p in self.vocab_probs.items()}
        token_doc_freq_json = {w: int(f) for w, f in self.token_doc_freq.items()}
        discriminative_scores_json = {w: float(s) for w, s in self.discriminative_scores.items()}
        
        config = {
            'beam_width': self.beam_width,
            'target_length': self.target_length,
            'use_semantic_heuristic': self.use_semantic_heuristic,
            'future_window': self.future_window,
            'future_eta': self.future_eta,
            'future_kappa': self.future_kappa,
            'retrieval_k': self.retrieval_k,
            'risk_alpha': self.risk_alpha,
            'risk_beta': self.risk_beta,
            'log_details': self.log_details,
            'base_weights': {k: float(v) for k, v in self.base_weights.items()},
            'stage_schedule': self.stage_schedule,
            'n_gram': self.n_gram,
        }
        
        (model_dir / 'config.json').write_text(json.dumps(config, indent=2))
        (model_dir / 'transitions.json').write_text(json.dumps(transitions_json, indent=2))
        (model_dir / 'context_counts.json').write_text(json.dumps(context_counts_json, indent=2))
        (model_dir / 'vocab_probs.json').write_text(json.dumps(vocab_probs_json, indent=2))
        (model_dir / 'token_doc_freq.json').write_text(json.dumps(token_doc_freq_json, indent=2))
        (model_dir / 'discriminative_scores.json').write_text(json.dumps(discriminative_scores_json, indent=2))
        
        # Save vectorizer with joblib (safe, only numpy arrays)
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, model_dir / 'vectorizer.joblib')
        
        print(f"Model saved to {model_dir}")
    
    @classmethod
    def load(cls, path: str) -> 'AStarBeamGenerator':
        """
        Load trained model from disk.
        
        Args:
            path: Directory path containing model files
        
        Returns:
            Loaded AStarBeamGenerator instance
        """
        import json
        import joblib
        from pathlib import Path
        
        model_dir = Path(path)
        
        config = json.loads((model_dir / 'config.json').read_text())
        
        # Load transitions
        transitions_json = json.loads((model_dir / 'transitions.json').read_text())
        context_counts_json = json.loads((model_dir / 'context_counts.json').read_text())
        vocab_probs_json = json.loads((model_dir / 'vocab_probs.json').read_text())
        token_doc_freq_json = json.loads((model_dir / 'token_doc_freq.json').read_text())
        discriminative_scores_json = json.loads((model_dir / 'discriminative_scores.json').read_text())
        
        # Reconstruct generator
        stage_schedule = config.pop('stage_schedule', [])
        base_weights = config.pop('base_weights', {})
        n_gram = config.pop('n_gram', 3)
        
        gen = cls(**config)
        gen.stage_schedule = stage_schedule
        gen.base_weights = base_weights
        gen.n_gram = n_gram
        
        # Reconstruct transitions (convert tuple strings back)
        import ast
        gen.transitions = defaultdict(Counter, {
            ast.literal_eval(ctx): Counter(probs)
            for ctx, probs in transitions_json.items()
        })
        
        gen.context_counts = Counter({
            ast.literal_eval(ctx): count
            for ctx, count in context_counts_json.items()
        })
        
        gen.vocab_probs = Counter(vocab_probs_json)
        gen.token_doc_freq = Counter(token_doc_freq_json)
        gen.discriminative_scores = discriminative_scores_json
        
        # Load vectorizer
        vectorizer_path = model_dir / 'vectorizer.joblib'
        if vectorizer_path.exists():
            gen.vectorizer = joblib.load(vectorizer_path)
        
        print(f"Model loaded from {model_dir}")
        return gen


class AStarAblationExperiment:
    """
    Ablation study framework for testing heuristic component impacts.
    """
    def __init__(self, training_comments: List[str], test_cases: List[Tuple[List[str], str, str]]):
        """
        Args:
            training_comments: Training data
            test_cases: List of (code_tokens, code_text, reference_comment) tuples
        """
        self.training_comments = training_comments
        self.test_cases = test_cases
        self.results = []
        
    def run_ablation(self, configs: List[Dict]) -> List[Dict]:
        """
        Run ablation study with different heuristic configurations.
        
        Args:
            configs: List of parameter dicts, each should have a 'name' key
        
        Returns:
            List of result dictionaries
        """
        print("\n" + "="*60)
        print("Running A* Heuristic Ablation Study")
        print("="*60)
        
        for config in configs:
            name = config.pop('name', 'unnamed')
            print(f"\nTesting configuration: {name}")
            print(f"  Parameters: {config}")
            
            # Create generator with this config
            gen = AStarBeamGenerator(**config)
            
            # Extract code texts for training TF-IDF
            code_texts = [code_text for _, code_text, _ in self.test_cases]
            
            # Train
            gen.train(self.training_comments, code_data=code_texts)
            
            # Evaluate on test cases
            results = self._evaluate_config(gen, name)
            results.update(config)  # Add config params to results
            results['config_name'] = name
            self.results.append(results)
            
            config['name'] = name  # Restore for next iteration
        
        return self.results
    
    def _evaluate_config(self, generator, config_name: str) -> Dict:
        """Evaluate a single configuration."""
        import time
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        bleu_scores = []
        runtimes = []
        smoothing = SmoothingFunction()
        
        for code_tokens, code_text, reference in self.test_cases:
            # Time generation
            start = time.time()
            generated = generator.generate(code_tokens, code_text, max_length=20)
            runtime = time.time() - start
            runtimes.append(runtime)
            
            # BLEU score
            ref_tokens = reference.lower().split()
            gen_tokens = generated.lower().split()
            bleu = sentence_bleu([ref_tokens], gen_tokens, 
                                smoothing_function=smoothing.method1)
            bleu_scores.append(bleu)
        
        return {
            'config_name': config_name,
            'bleu_mean': np.mean(bleu_scores),
            'bleu_std': np.std(bleu_scores),
            'runtime_mean_ms': np.mean(runtimes) * 1000,
            'runtime_std_ms': np.std(runtimes) * 1000,
        }


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
    
    code_samples = [
        "def sum(a, b): return a + b",
        "def add(x, y): return x + y",
        "def calculate(num1, num2): return num1 + num2",
        "def compute(parameter1, parameter2): return parameter1 + parameter2",
        "def add_numbers(x, y): return x + y",
    ]
    
    # Train model
    gen = AStarBeamGenerator(beam_width=5, alpha=0.2, beta=0.4, gamma=0.4)
    gen.train(training_comments, code_data=code_samples)
    
    # Generate comment
    code_tokens = ["sum", "add", "numbers"]
    code_text = "def sum(a, b): return a + b"
    comment = gen.generate(code_tokens, code_text, max_length=15)
    
    print(f"\nGenerated: {comment}")
    print(f"  Expected: something like 'calculate sum of numbers and return'")
    
    # Example ablation study
    print("\n" + "="*60)
    print("ABLATION STUDY EXAMPLE")
    print("="*60)
    
    test_cases = [
        (["sum", "add"], "def sum(a, b): return a + b", "calculate sum of two numbers"),
        (["compute"], "def compute(x): return x", "compute the value"),
    ]
    
    ablation = AStarAblationExperiment(training_comments, test_cases)
    
    configs = [
        {'name': 'full_heuristic', 'alpha': 0.2, 'beta': 0.4, 'gamma': 0.4, 'use_semantic_heuristic': True},
        {'name': 'no_semantic', 'alpha': 0.2, 'beta': 0.4, 'gamma': 0.4, 'use_semantic_heuristic': False},
        {'name': 'no_coverage', 'alpha': 0.5, 'beta': 0.0, 'gamma': 0.5, 'use_semantic_heuristic': True},
        {'name': 'baseline_length_only', 'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0, 'use_semantic_heuristic': False},
    ]
    
    results = ablation.run_ablation(configs)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['config_name']:25s} | BLEU: {r['bleu_mean']:.3f} +/- {r['bleu_std']:.3f} | "
              f"Runtime: {r['runtime_mean_ms']:.2f}ms")
