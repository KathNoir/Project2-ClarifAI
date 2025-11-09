"""
Startup script for the backend API server.
Run this to start the FastAPI server.
"""
import uvicorn
from pathlib import Path
import sys

# Add backend to path
BACKEND_ROOT = Path(__file__).parent
sys.path.insert(0, str(BACKEND_ROOT))

from config import BACKEND_PORT, BACKEND_HOST, API_URL

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Comment Generation API Server")
    print("=" * 60)
    print(f"Server will be available at: {API_URL}")
    print(f"API docs: {API_URL}/docs")
    print(f"Health check: {API_URL}/")
    print("=" * 60)
    
    uvicorn.run(
        "api.server:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True,
        log_level="info"
    )

