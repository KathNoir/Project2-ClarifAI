"""
Test the backend API.
This script tests the /generate endpoint without requiring trained models.
"""
import sys
from pathlib import Path

# Add backend to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import requests
import json
from config import API_URL

def test_health():
    """Test health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("  ERROR: Cannot connect to API. Is the server running?")
        print("  Start server with: python run_server.py")
        return False

def test_generate():
    """Test comment generation endpoint."""
    print("\nTesting /generate endpoint...")
    
    test_cases = [
        {
            "code": "def sum(a, b):\n    return a + b",
            "max_length": 20
        },
        {
            "code": "def multiply(x, y):\n    result = x * y\n    return result",
            "max_length": 20
        },
        {
            "code": "def find_max(arr):\n    if not arr:\n        return None\n    return max(arr)",
            "max_length": 25
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Test {i}:")
        print(f"  Code:\n{test_case['code']}")
        
        try:
            response = requests.post(
                f"{API_URL}/generate",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Success!")
                print(f"  Greedy: {result.get('greedy_comment', 'N/A')} ({result.get('greedy_runtime_ms', 0):.2f}ms)")
                print(f"  A*: {result.get('astar_comment', 'N/A')} ({result.get('astar_runtime_ms', 0):.2f}ms)")
                print(f"  Tokens: {result.get('code_tokens', [])}")
            else:
                print(f"  ERROR: Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            print("  ERROR: Cannot connect to API. Is the server running?")
            return False
        except Exception as e:
            print(f"  ERROR: Error: {e}")
            return False
    
    return True

def test_models_endpoint():
    """Test /models endpoint."""
    print("\nTesting /models endpoint...")
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            models = response.json()
            print(f"  Greedy loaded: {models['greedy']['loaded']}")
            print(f"  A* loaded: {models['astar']['loaded']}")
        return True
    except Exception as e:
        print(f"  ERROR: Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Backend API Test")
    print("=" * 60)
    
    if not test_health():
        print("\nWARNING: Make sure to start the server first:")
        print("  cd backend")
        print("  python run_server.py")
        exit(1)
    
    test_models_endpoint()
    test_generate()
    
    print("\n" + "=" * 60)
    print("API Test Complete!")
    print("=" * 60)

