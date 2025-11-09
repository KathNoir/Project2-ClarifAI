"""
Complete system test - tests A* service integration with API.
This creates a temporary model to test the full pipeline.
"""
import sys
from pathlib import Path

# Add backend to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from services.code_processor import extract_code_tokens
from services.astar_service import AStarService
from algos.A_star_beam import AStarBeamGenerator

def create_test_model():
    """Create a quick test model for A* service."""
    print("=" * 60)
    print("Creating Test A* Model")
    print("=" * 60)
    
    # Training data
    training_comments = [
        "calculate the sum of two numbers",
        "return the sum of a and b",
        "add two integers and return result",
        "compute sum of input parameters",
        "function to add numbers together",
        "multiply two numbers and return product",
        "divide first number by second number",
        "subtract second number from first number",
        "find maximum value in array",
        "sort array in ascending order",
        "check if number is even",
        "count number of elements in list",
        "reverse the order of list items",
        "concatenate two strings together",
        "convert string to uppercase letters",
    ]
    
    code_samples = [
        "def sum(a, b): return a + b",
        "def add(x, y): return x + y",
        "def calculate(num1, num2): return num1 + num2",
        "def multiply(x, y): return x * y",
        "def divide(a, b): return a / b",
        "def subtract(a, b): return a - b",
        "def max(arr): return max(arr)",
        "def sort(items): return sorted(items)",
        "def is_even(n): return n % 2 == 0",
        "def count(lst): return len(lst)",
    ]
    
    # Create model directory
    model_dir = BACKEND_ROOT / "models" / "astar_A3_dependency"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Creating A* generator with A3_dependency config...")
    generator = AStarBeamGenerator(
        beam_width=5,
        alpha=0.2,
        beta=0.4,
        gamma=0.4,
        target_length=10,
        use_semantic_heuristic=True,
        lambda_future=0.0,
        lambda_contrastive=0.0,
        lambda_dependency=0.4,  # A3_dependency variant
        lambda_risk=0.0,
    )
    
    print("2. Training model...")
    generator.train(training_comments, code_data=code_samples, n_gram=3)
    
    print("3. Saving model...")
    generator.save(str(model_dir))
    
    print(f"   Model saved to {model_dir}")
    print("\nTest model created! Now you can test the API.")
    return model_dir

def test_service():
    """Test A* service with the model."""
    print("\n" + "=" * 60)
    print("Testing A* Service Integration")
    print("=" * 60)
    
    service = AStarService()
    
    if not service.is_loaded():
        print("\nWARNING: Model not loaded. Creating test model...")
        create_test_model()
        service = AStarService()  # Reload
    
    if service.is_loaded():
        print("\nService loaded successfully!")
        
        test_code = "def sum(a, b):\n    return a + b"
        tokens = extract_code_tokens(test_code)
        
        print(f"\nTesting with code: {test_code}")
        print(f"Extracted tokens: {tokens}")
        
        comment, runtime = service.generate(
            code_tokens=tokens,
            code_text=test_code,
            max_length=20
        )
        
        print(f"\nGenerated comment: '{comment}'")
        print(f"Runtime: {runtime:.2f}ms")
        print("\nService integration working!")
        return True
    else:
        print("ERROR: Service failed to load model")
        return False

if __name__ == "__main__":
    print("\nFull System Test")
    print("=" * 60)
    
    # Test 1: Create model if needed
    model_path = BACKEND_ROOT / "models" / "astar_A3_dependency"
    if not model_path.exists():
        print("\nNo model found. Creating test model...")
        create_test_model()
    
    # Test 2: Test service
    success = test_service()
    
    if success:
        print("\n" + "=" * 60)
        print("Ready for API testing!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start API server: python run_server.py")
        print("2. In another terminal, test: python tests/test_api.py")
        print("3. Or visit: http://localhost:8000/docs")
    else:
        print("\nERROR: Setup incomplete. Check errors above.")

