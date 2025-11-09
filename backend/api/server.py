"""
FastAPI Server - Main API Endpoint

Handles requests from frontend and generates comments using both algorithms.

Team Responsibilities:
- Katherine: Model loading (in services/)
- Jarrod: Greedy service (services/greedy_service.py)
- Jennifer: A* service (services/astar_service.py)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

# Add backend to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from api.models import GenerateRequest, GenerateResponse
from services.code_processor import extract_code_tokens, validate_python_code
from services.greedy_service import GreedyService  # Jarrod's area
from services.astar_service import AStarService   # Jennifer's area
from config import FRONTEND_URLS, DEFAULT_MAX_LENGTH

app = FastAPI(
    title="Comment Generation API",
    description="Generate code comments using Greedy and A* algorithms",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (Katherine loads models, Jarrod/Jennifer implement services)
greedy_service = GreedyService()  # Jarrod's implementation
astar_service = AStarService()    # Jennifer's implementation


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "greedy_loaded": greedy_service.is_loaded(),
        "astar_loaded": astar_service.is_loaded(),
        "ready": greedy_service.is_loaded() and astar_service.is_loaded()
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_comments(request: GenerateRequest):
    """
    Generate comments for Python code using both algorithms.
    
    Args:
        request: Contains Python code string
        
    Returns:
        Both generated comments, tokens, and metrics
    """
    # Validate code (Katherine's implementation)
    try:
        is_valid, error_msg = validate_python_code(request.code)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg or "Invalid Python code")
    except NotImplementedError:
        raise HTTPException(
            status_code=503,
            detail="Code validation not yet implemented. Katherine needs to implement validate_python_code() in services/code_processor.py"
        )
    
    # Extract tokens (Katherine's implementation)
    try:
        code_tokens = extract_code_tokens(request.code)
    except NotImplementedError:
        raise HTTPException(
            status_code=503,
            detail="Code token extraction not yet implemented. Katherine needs to implement extract_code_tokens() in services/code_processor.py"
        )
    
    if not code_tokens:
        raise HTTPException(
            status_code=400, 
            detail="Could not extract meaningful tokens from code. Make sure your code has function names, variables, etc."
        )
    
    # Generate comments using both services
    # Note: Services will raise errors if models not loaded, but we catch them gracefully
    greedy_steps = None
    astar_steps = None
    
    if not greedy_service.is_loaded():
        greedy_comment = "[Greedy model not loaded]"
        greedy_runtime = 0.0
    else:
        try:
            result = greedy_service.generate(
                code_tokens=code_tokens,
                max_length=request.max_length,
                enable_visualization=request.enable_visualization
            )
            greedy_comment, greedy_runtime, greedy_steps = result
        except Exception as e:
            greedy_comment = f"[Error generating comment: {str(e)}]"
            greedy_runtime = 0.0
    
    if not astar_service.is_loaded():
        astar_comment = "[A* model not loaded]"
        astar_runtime = 0.0
    else:
        try:
            result = astar_service.generate(
                code_tokens=code_tokens,
                code_text=request.code,
                max_length=request.max_length,
                enable_visualization=request.enable_visualization
            )
            astar_comment, astar_runtime, astar_steps = result
        except Exception as e:
            astar_comment = f"[Error generating comment: {str(e)}]"
            astar_runtime = 0.0
    
    # Build response
    result = {
        "greedy_comment": greedy_comment,
        "astar_comment": astar_comment,
        "code_tokens": code_tokens,
        "greedy_runtime_ms": greedy_runtime,
        "astar_runtime_ms": astar_runtime,
        "greedy_loaded": greedy_service.is_loaded(),
        "astar_loaded": astar_service.is_loaded(),
        "error": None,
        "greedy_steps": greedy_steps,
        "astar_steps": astar_steps,
    }
    
    return GenerateResponse(**result)


@app.get("/models")
async def list_models():
    """Check model loading status."""
    return {
        "greedy": {
            "loaded": greedy_service.is_loaded(),
            "path": str(greedy_service.model_path),
            "variant": "G0_base"
        },
        "astar": {
            "loaded": astar_service.is_loaded(),
            "path": str(astar_service.model_path),
            "variant": "A3_dependency"
        }
    }


if __name__ == "__main__":
    import uvicorn
    from config import BACKEND_HOST, BACKEND_PORT
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)

