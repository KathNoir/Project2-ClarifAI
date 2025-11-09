"""
Configuration constants for the backend API.
Centralizes hardcoded values for easier maintenance.
"""
from pathlib import Path

# Server Configuration
BACKEND_PORT = 8001
BACKEND_HOST = "0.0.0.0"

# Frontend Configuration
FRONTEND_PORT = 3000
FRONTEND_URLS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8000",  # mkdocs
]

# API Configuration
API_URL = f"http://localhost:{BACKEND_PORT}"

# Model Configuration
MODELS_DIR = Path(__file__).parent / "models"
GREEDY_MODEL_NAME = "greedy_G0_base"
ASTAR_MODEL_NAME = "astar_A3_dependency"

# Default Generation Parameters
DEFAULT_MAX_LENGTH = 20
MAX_TOKENS = 20

