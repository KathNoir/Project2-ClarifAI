"""
API Request/Response Models
"""
from pydantic import BaseModel
from typing import Optional


class GenerateRequest(BaseModel):
    """Request model for comment generation."""
    code: str
    max_length: int = 20
    enable_visualization: bool = False


class GenerateResponse(BaseModel):
    """Response model with both generated comments."""
    greedy_comment: str
    astar_comment: str
    code_tokens: list[str]
    greedy_runtime_ms: float
    astar_runtime_ms: float
    greedy_loaded: bool
    astar_loaded: bool
    error: Optional[str] = None
    greedy_steps: Optional[list[dict]] = None
    astar_steps: Optional[list[dict]] = None

