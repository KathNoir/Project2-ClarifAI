"""
Quick API test - simple one-shot test.
Run this while the server is running.
"""
import sys
from pathlib import Path

# Add backend to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import requests
import json
from config import API_URL

def quick_test():
    """Quick test of the /generate endpoint."""
    print("=" * 60)
    print("Quick API Test")
    print("=" * 60)
    
    url = f"{API_URL}/generate"
    test_code = {
        "code": "def sum(a, b):\n    return a + b",
        "max_length": 20
    }
    
    print(f"\nTesting: {test_code['code']}")
    print(f"Endpoint: {url}\n")
    
    try:
        response = requests.post(url, json=test_code)
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print(f"\nA* Comment: {result['astar_comment']}")
            print(f"A* Runtime: {result['astar_runtime_ms']:.2f}ms")
            print(f"Greedy: {result['greedy_comment']}")
            print(f"Tokens: {result['code_tokens']}")
            print("\n" + "=" * 60)
            print("API is working! A* is generating comments!")
        else:
            print(f"ERROR: Error {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server!")
        print("Make sure you ran: python run_server.py")
    except Exception as e:
        print(f"ERROR: Error: {e}")

if __name__ == "__main__":
    quick_test()

