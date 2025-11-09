"""
Unified test runner for the Code Comment Generation project.
Run all tests or specific test suites.
"""
import sys
import argparse
from pathlib import Path

# Add backend to path
BACKEND_ROOT = Path(__file__).parent / "backend"
TESTS_DIR = BACKEND_ROOT / "tests"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(TESTS_DIR))

def run_api_tests():
    """Run API endpoint tests."""
    print("=" * 60)
    print("Testing API Endpoints")
    print("=" * 60)
    try:
        # Import test_api module
        import test_api as test_api_module
        
        if not test_api_module.test_health():
            print("\nWARNING: Make sure to start the server first:")
            print("  cd backend")
            print("  python run_server.py")
            return False
        
        test_api_module.test_models_endpoint()
        test_api_module.test_generate()
        
        print("\n" + "=" * 60)
        print("API Tests Complete!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"ERROR: API test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_astar_tests():
    """Run A* algorithm tests."""
    print("=" * 60)
    print("Testing A* Algorithm")
    print("=" * 60)
    try:
        import test_astar
        test_astar.test_astar()
        return True
    except Exception as e:
        print(f"ERROR: A* test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_service_tests():
    """Run service integration tests."""
    print("=" * 60)
    print("Testing Service Integration")
    print("=" * 60)
    try:
        import test_full_system
        # Run the test which creates model and tests service
        model_path = BACKEND_ROOT / "models" / "astar_A3_dependency"
        if not model_path.exists():
            print("\nNo model found. Creating test model...")
            test_full_system.create_test_model()
        
        success = test_full_system.test_service()
        return success
    except Exception as e:
        print(f"ERROR: Service test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_tests():
    """Run quick smoke tests."""
    print("=" * 60)
    print("Quick API Smoke Test")
    print("=" * 60)
    try:
        import test_quick
        test_quick.quick_test()
        return True
    except Exception as e:
        print(f"ERROR: Quick test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Unified test runner")
    parser.add_argument(
        "--suite",
        choices=["api", "astar", "service", "quick", "all"],
        default="all",
        help="Test suite to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Change to backend directory for imports
    import os
    original_dir = os.getcwd()
    os.chdir(BACKEND_ROOT)
    
    try:
        results = {}
        
        if args.suite == "all":
            print("\nRunning All Test Suites\n")
            results["astar"] = run_astar_tests()
            print("\n")
            results["service"] = run_service_tests()
            print("\n")
            results["api"] = run_api_tests()
            print("\n")
            results["quick"] = run_quick_tests()
        elif args.suite == "api":
            results["api"] = run_api_tests()
        elif args.suite == "astar":
            results["astar"] = run_astar_tests()
        elif args.suite == "service":
            results["service"] = run_service_tests()
        elif args.suite == "quick":
            results["quick"] = run_quick_tests()
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        for suite, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {suite:10} {status}")
        
        all_passed = all(results.values())
        if all_passed:
            print("\nAll tests passed!")
        else:
            print("\nWARNING: Some tests failed. Check output above.")
        
        return 0 if all_passed else 1
    
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    sys.exit(main())

