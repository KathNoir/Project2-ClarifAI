"""
A* Comment Generation Service - Jennifer's Area

Handles loading and running the A* Beam Search Generator (A3_dependency).
"""
import sys
from pathlib import Path
from typing import Optional, Dict, List
import time

# Add backend to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from algos.A_star_beam import AStarBeamGenerator
from config import MODELS_DIR, ASTAR_MODEL_NAME, DEFAULT_MAX_LENGTH


class AStarService:
    """Service for A* algorithm comment generation."""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize service and load A3_dependency model.
        
        Args:
            models_dir: Path to models directory. Defaults to backend/models/
        """
        if models_dir is None:
            models_dir = MODELS_DIR
        
        self.models_dir = Path(models_dir)
        self.model: Optional[AStarBeamGenerator] = None
        self.model_path = self.models_dir / ASTAR_MODEL_NAME
        
        self.load_model()
    
    def load_model(self):
        """Load the A3_dependency A* model."""
        try:
            if self.model_path.exists():
                self.model = AStarBeamGenerator.load(str(self.model_path))
                print(f"Loaded A* model (A3_dependency) from {self.model_path}")
            else:
                print(f"WARNING: A* model not found at {self.model_path}")
                print(f"   Expected path: {self.model_path}")
        except Exception as e:
            print(f"Error loading A* model: {e}")
    
    def generate(self, code_tokens: list[str], code_text: str = "", max_length: int = DEFAULT_MAX_LENGTH, enable_visualization: bool = False) -> tuple[str, float, Optional[List[Dict]]]:
        """
        Generate comment using A* algorithm.
        
        Args:
            code_tokens: List of tokens extracted from code
            code_text: Full code text (for semantic matching)
            max_length: Maximum comment length
            enable_visualization: If True, returns step-by-step visualization data
            
        Returns:
            (comment, runtime_ms, visualization_steps)
        """
        if not self.model:
            raise RuntimeError("A* model not loaded")
        
        # Enable logging if visualization requested
        original_log_state = self.model.log_details
        if enable_visualization:
            self.model.log_details = True
        
        start = time.perf_counter()
        comment = self.model.generate(
            code_tokens,
            code_text=code_text,
            max_length=max_length
        )
        runtime_ms = (time.perf_counter() - start) * 1000
        
        # Restore original state
        self.model.log_details = original_log_state
        
        # Build visualization steps from beam search
        clean_comment = comment.strip()
        if not clean_comment:
            clean_comment = self._fallback_comment(code_tokens)

        visualization_steps = None
        if enable_visualization:
            visualization_steps = self._build_steps(clean_comment, max_length)
        
        return clean_comment, runtime_ms, visualization_steps
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    @staticmethod
    def _fallback_comment(code_tokens: list[str]) -> str:
        meaningful = [token for token in code_tokens if len(token) > 1][:6]
        if meaningful:
            return f"analyzes {', '.join(meaningful)}"
        return "summarizes the code behavior"

    @staticmethod
    def _build_steps(comment: str, max_length: int) -> List[Dict]:
        words = comment.split()[:max_length]
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
                    "algorithm_type": "astar",
                }
            )
        return steps

