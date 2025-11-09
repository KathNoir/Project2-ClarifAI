"""
Test script for A* algorithm - works without trained models.
Trains a quick model on sample data and tests generation.
"""
import sys
from pathlib import Path

# Add backend to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from algos.A_star_beam import AStarBeamGenerator
from services.code_processor import extract_code_tokens

def test_astar():
    """Test A* algorithm with a minimal trained model."""
    
    print("=" * 60)
    print("Testing A* Algorithm")
    print("=" * 60)
    
    # Sample training data
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
    
    print("\n1. Creating A* generator...")
    generator = AStarBeamGenerator(
        beam_width=5,
        alpha=0.2,  # length bias
        beta=0.4,   # coverage
        gamma=0.4,  # semantic
        target_length=10,
        use_semantic_heuristic=True,
        lambda_future=0.0,
        lambda_contrastive=0.0,
        lambda_dependency=0.4,  # A3_dependency variant
        lambda_risk=0.0,
    )
    
    print("2. Training model on sample data...")
    generator.train(training_comments, code_data=code_samples, n_gram=3)
    print("   Training complete!")
    
    print("\n3. Testing code token extraction...")
    test_code = "def sum(a, b):\n    return a + b"
    tokens = extract_code_tokens(test_code)
    print(f"   Code: {test_code}")
    print(f"   Extracted tokens: {tokens}")
    
    print("\n4. Generating comment with A*...")
    import time
    start_time = time.perf_counter()
    comment = generator.generate(
        code_tokens=tokens,
        code_text=test_code,
        max_length=20
    )
    runtime_ms = (time.perf_counter() - start_time) * 1000
    
    print(f"   Generated comment: '{comment}'")
    print(f"   Runtime: {runtime_ms:.2f}ms")
    
    print("\n5. Testing with more examples...")
    test_cases = [
        ("def multiply(x, y): return x * y", ["multiply", "x", "y"]),
        ("def find_max(arr): return max(arr)", ["find_max", "arr", "max"]),
        ("def is_even(n): return n % 2 == 0", ["is_even", "n"]),
    ]
    
    for code, expected_tokens in test_cases:
        tokens = extract_code_tokens(code)
        start_time = time.perf_counter()
        comment = generator.generate(tokens, code_text=code, max_length=15)
        runtime_ms = (time.perf_counter() - start_time) * 1000
        print(f"   Code: {code}")
        print(f"   Comment: '{comment}' ({runtime_ms:.2f}ms)")
        print()
    
    print("=" * 60)
    print("A* Algorithm Test Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test with backend API: python -m backend.api.server")
    print("2. Visit http://localhost:8000/docs for API documentation")
    print("3. Use POST /generate endpoint with your code")

if __name__ == "__main__":
    test_astar()

