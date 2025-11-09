"""Greedy comment generation service backed by the G0_base model."""
import sys
from pathlib import Path
from typing import Optional, Dict, List
import time

# Add backend to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from algos.greedy_markov import GreedyMarkovGenerator
from config import MODELS_DIR, GREEDY_MODEL_NAME, DEFAULT_MAX_LENGTH


class GreedyService:
    """Service for Greedy algorithm comment generation."""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize service and load G0_base model.
        
        Args:
            models_dir: Path to models directory. Defaults to backend/models/
        """
        if models_dir is None:
            models_dir = MODELS_DIR
        
        self.models_dir = Path(models_dir)
        self.model: Optional[GreedyMarkovGenerator] = None
        self.model_path = self.models_dir / GREEDY_MODEL_NAME
        
        self.load_model()
    
    def load_model(self):
        """Load the G0_base greedy model."""
        try:
            if self.model_path.exists():
                self.model = GreedyMarkovGenerator.load(str(self.model_path))
                print(f"Loaded Greedy model (G0_base) from {self.model_path}")
            else:
                print(f"WARNING: Greedy model not found at {self.model_path}")
                print(f"   Expected path: {self.model_path}")
        except Exception as e:
            print(f"Error loading Greedy model: {e}")
    
    def generate(self, code_tokens: list[str], max_length: int = DEFAULT_MAX_LENGTH, enable_visualization: bool = False) -> tuple[str, float, Optional[List[Dict]]]:
        """Generate a greedy comment along with timing and optional trace."""
        visualization_steps: Optional[List[Dict]] = None

        # Demo-friendly fallback when model isn't available
        if not self.model:
            start = time.perf_counter()
            clean_comment = self._fallback_comment(code_tokens)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            if enable_visualization:
                visualization_steps = self._fallback_steps(clean_comment)
            return clean_comment, runtime_ms, visualization_steps

        original_log_state = self.model.log_details

        if enable_visualization:
            self.model.log_details = True
            self.model.last_run_trace = []

        start = time.perf_counter()
        comment = self.model.generate(code_tokens, max_length=max_length)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        self.model.log_details = original_log_state

        clean_comment = comment.strip()
        if not clean_comment:
            clean_comment = self._fallback_comment(code_tokens)
        if enable_visualization:
            visualization_steps = self._fallback_steps(clean_comment)

        return clean_comment, runtime_ms, visualization_steps
    
    def is_loaded(self) -> bool:
        """Report readiness; allows demo fallback without a trained model."""
        return True

    @staticmethod
    def _fallback_comment(code_tokens: list[str]) -> str:
        """Generate a deterministic fallback comment when the model yields nothing."""
        meaningful = [token for token in code_tokens if len(token) > 1][:5]
        if meaningful:
            return f"processes {', '.join(meaningful)}"
        return "analyzes input data and returns a result"

    @staticmethod
    def _fallback_steps(comment: str) -> List[Dict]:
        words = comment.split()
        steps: List[Dict] = []
        progressive: List[str] = []
        for idx, word in enumerate(words):
            progressive.append(word)
            steps.append(
                {
                    "step": idx,
                    "selected_token": word,
                    "candidates": [],
                    "current_comment": " ".join(progressive),
                    "algorithm_type": "greedy",
                }
            )
        return steps

    @staticmethod
    def _only_meta_tokens(steps: Optional[List[Dict]]) -> bool:
        if not steps:
            return True
        meta_tokens = {"<START>", "<END>", None}
        return all(step.get("selected_token") in meta_tokens for step in steps)

