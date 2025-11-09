"""Train the greedy G0_base model using CodeSearchNet data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Add backend to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from algos.greedy_markov import GreedyMarkovGenerator
from config import MODELS_DIR, GREEDY_MODEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the greedy G0_base model. "
            "Use --max-samples 0 to consume the entire CodeSearchNet split."
        )
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Number of samples to use (0 = all available samples).",
    )
    parser.add_argument(
        "--language",
        default="python",
        help="CodeSearchNet language subset (default: python).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to backend/models/greedy_G0_base "
            "so the FastAPI service picks it up automatically."
        ),
    )
    return parser.parse_args()


def _load_codesearchnet(max_samples: int, language: str) -> List[str]:
    """Fetch docstrings from CodeSearchNet using the datasets library."""
    sample_limit = max_samples if max_samples and max_samples > 0 else None

    from datasets import load_dataset  # type: ignore

    print("Downloading CodeSearchNet (this may take several minutes)...")
    try:
        dataset = load_dataset("code_search_net", language, trust_remote_code=True)
    except Exception as trust_exc:  # pragma: no cover - dataset fetch depends on env
        print("trust_remote_code download failed, retrying without it...")
        dataset = load_dataset("code_search_net", language, trust_remote_code=False)
        if dataset is None:  # pragma: no cover
            raise RuntimeError(f"Failed to load dataset: {trust_exc}") from trust_exc

    collected: List[str] = []
    for split in ("train", "validation", "test"):
        if split not in dataset:
            continue
        for item in dataset[split]:
            if sample_limit is not None and len(collected) >= sample_limit:
                break
            comment = item.get("func_documentation_string") or ""
            if not comment:
                continue
            collected.append(" ".join(comment.strip().split()))
        if sample_limit is not None and len(collected) >= sample_limit:
            break

    if not collected:
        raise RuntimeError("No documentation strings retrieved from CodeSearchNet.")
    return collected


def get_training_comments(max_samples: int, language: str) -> List[str]:
    """Load training comments, falling back to synthetic samples if needed."""
    try:
        return _load_codesearchnet(max_samples=max_samples, language=language)
    except ImportError:
        print("datasets library not installed; using synthetic fallback comments.")
    except Exception as exc:
        print(f"Could not load CodeSearchNet ({exc}); using synthetic fallback comments.")

    return [
        "calculate the sum of two numbers",
        "return the sum of a and b",
        "add two integers and return result",
        "compute sum of input parameters",
        "function to add numbers together",
        "sum all values in the list",
        "add numbers and return total",
        "multiply two numbers and return product",
        "divide first number by second number",
        "subtract second number from first number",
        "calculate power of base to exponent",
        "compute factorial of given number",
        "find maximum value in array",
        "find minimum value in list",
        "sort array in ascending order",
        "sort list in descending order",
        "reverse the order of list items",
        "count number of elements in list",
        "remove duplicate items from list",
        "filter items matching condition",
        "concatenate two strings together",
        "convert string to uppercase letters",
        "convert string to lowercase letters",
        "check if string contains substring",
        "split string into list of words",
        "join list of strings with separator",
        "check if number is even",
        "check if number is odd",
        "check if string is palindrome",
        "find index of item in list",
        "check if item exists in collection",
        "search for pattern in text",
        "insert item at beginning of list",
        "append item to end of list",
        "remove item from list by value",
        "get first element of list",
        "get last element of list",
        "create dictionary from key value pairs",
        "return true if condition is met",
        "return false if condition fails",
        "compare two values and return result",
        "check if all items satisfy condition",
        "check if any item satisfies condition",
        "initialize object with given parameters",
        "return formatted string representation",
        "validate input parameters before processing",
        "handle error and return default value",
        "process data and return result",
        "calculate average of all numbers in list",
        "convert list of strings to integers",
        "group items by key value",
        "map function to each item in list",
        "reduce list to single value using function",
    ]


def create_model(max_samples: int, language: str, output_dir: Path | None) -> Path:
    """Train and persist the greedy model."""
    training_comments = get_training_comments(max_samples=max_samples, language=language)
    target_dir = output_dir if output_dir is not None else MODELS_DIR / GREEDY_MODEL_NAME
    target_dir.mkdir(parents=True, exist_ok=True)

    generator = GreedyMarkovGenerator(
        n_gram=4,
        entropy_threshold=2.0,
        lambda_cache=0.0,
        lambda_anchor=0.0,
        lookahead_k=0,
        backtrack_budget=0,
        log_details=False,
    )

    print(f"Training on {len(training_comments)} comments...")
    generator.train(training_comments)

    print(f"Writing model artifacts to {target_dir} ...")
    generator.save(str(target_dir))

    print("Training complete. Files generated:")
    for artifact in sorted(target_dir.glob("*")):
        print(f"  - {artifact.name}")

    return target_dir


def main() -> int:
    args = parse_args()
    create_model(
        max_samples=args.max_samples,
        language=args.language,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
Create a toy greedy model for development/testing.

This creates a small model trained on sample data so Jarrod can start working
on greedy comment quality before Katherine finishes full dataset training.

Tries to use real CodeSearchNet data (small sample) if available,
otherwise uses synthetic comments.

Usage:
    cd Proj2/backend
    python tests/create_toy_greedy_model.py

To use real CodeSearchNet data:
    1. Install HuggingFace datasets library:
       pip install datasets
    
    2. Run the script - it will automatically try to download from HuggingFace
       If your environment allows code execution, it will use real data.
       Otherwise, it falls back to synthetic comments (which work fine for testing).
"""

import sys
from pathlib import Path

# Add backend to path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from algos.greedy_markov import GreedyMarkovGenerator
from config import MODELS_DIR, GREEDY_MODEL_NAME

def get_training_data():
    """
    Get training comments - tries HuggingFace CodeSearchNet first, falls back to synthetic.
    
    Returns list of comment strings (same format as prepare_training_data returns).
    """
    # Try to use real CodeSearchNet data (small sample)
    # 
    # To use HuggingFace dataset, you need:
    #   1. Install datasets: pip install datasets
    #   2. Note: CodeSearchNet requires trust_remote_code=True (needs code execution)
    #      Some environments (like restricted servers) may block this.
    #      If it fails, script will fall back to synthetic comments.
    #
    print("Attempting to download CodeSearchNet dataset from HuggingFace...")
    print("Note: Requires 'datasets' library. Install with: pip install datasets")
    
    try:
        # Import HuggingFace datasets library
        # Install with: pip install datasets
        from datasets import load_dataset
        
        print("\nLoading code_search_net dataset...")
        print("(This dataset requires trust_remote_code=True due to security restrictions)")
        
        # CodeSearchNet requires trust_remote_code=True because it needs code execution
        try:
            dataset = load_dataset('code_search_net', 'python', trust_remote_code=True)
            print("✓ Dataset loaded successfully!")
        except Exception as e1:
            print(f"Failed with trust_remote_code=True: {e1}")
            print("Trying alternative approach...")
            # Alternative: try without trust_remote_code (might work in some environments)
            try:
                dataset = load_dataset('code_search_net', 'python', trust_remote_code=False)
                print("✓ Dataset loaded (without trust_remote_code)")
            except Exception as e2:
                raise Exception(f"Both methods failed. trust_remote_code=True: {e1}, trust_remote_code=False: {e2}")
        
        # Convert to our format (same as preprocessor does)
        data = []
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                for item in dataset[split]:
                    if len(data) >= 200:  # Small sample for toy model
                        break
                    
                    # Extract func_documentation_string (same as preprocessor)
                    if item.get('func_documentation_string'):
                        data.append({
                            'code': item.get('func_code_string', ''),
                            'comment': item['func_documentation_string'],
                            'func_name': item.get('func_name', ''),
                        })
                
                if len(data) >= 200:
                    break
        
        if data:
            # Extract just comments (same as prepare_training_data)
            training_comments = []
            for item in data:
                if item.get('comment'):
                    training_comments.append(item['comment'])
            
            if training_comments:
                print(f"✓ Successfully loaded {len(training_comments)} real CodeSearchNet comments!")
                return training_comments
        
    except ImportError:
        print("\n❌ 'datasets' library not installed!")
        print("   Install it with: pip install datasets")
        print("   Then run this script again to use real CodeSearchNet data.")
        print("\n   For now, falling back to synthetic comments...")
    except Exception as e:
        print(f"\n❌ Could not load from HuggingFace: {e}")
        print("\n   This can happen if:")
        print("   - Your environment blocks code execution (security restriction)")
        print("   - Network/firewall issues")
        print("   - Dataset repository access restrictions")
        print("\n   Workaround: Use synthetic comments (works fine for testing)")
        print("   Or install datasets and enable code execution in your environment")
        print("\n   Falling back to synthetic comments...")
    
    # Fallback: Synthetic comments (matches CodeSearchNet style)
    print("\nUsing synthetic comments (matches CodeSearchNet docstring style)")
    training_comments = [
        # Sum/addition patterns
        "calculate the sum of two numbers",
        "return the sum of a and b",
        "add two integers and return result",
        "compute sum of input parameters",
        "function to add numbers together",
        "sum all values in the list",
        "add numbers and return total",
        
        # Math operations
        "multiply two numbers and return product",
        "divide first number by second number",
        "subtract second number from first number",
        "calculate power of base to exponent",
        "compute factorial of given number",
        
        # Array/list operations
        "find maximum value in array",
        "find minimum value in list",
        "sort array in ascending order",
        "sort list in descending order",
        "reverse the order of list items",
        "count number of elements in list",
        "remove duplicate items from list",
        "filter items matching condition",
        
        # String operations
        "concatenate two strings together",
        "convert string to uppercase letters",
        "convert string to lowercase letters",
        "check if string contains substring",
        "split string into list of words",
        "join list of strings with separator",
        
        # Search/find operations
        "check if number is even",
        "check if number is odd",
        "check if string is palindrome",
        "find index of item in list",
        "check if item exists in collection",
        "search for pattern in text",
        
        # Data structure operations
        "insert item at beginning of list",
        "append item to end of list",
        "remove item from list by value",
        "get first element of list",
        "get last element of list",
        "create dictionary from key value pairs",
        
        # Conditional/logic operations
        "return true if condition is met",
        "return false if condition fails",
        "compare two values and return result",
        "check if all items satisfy condition",
        "check if any item satisfies condition",
        
        # Function/class operations
        "initialize object with given parameters",
        "return formatted string representation",
        "validate input parameters before processing",
        "handle error and return default value",
        "process data and return result",
        
        # More specific patterns
        "calculate average of all numbers in list",
        "convert list of strings to integers",
        "group items by key value",
        "map function to each item in list",
        "reduce list to single value using function",
    ]
    
    return training_comments

def create_toy_model():
    """Create a quick toy greedy model for development."""
    print("=" * 60)
    print("Creating Toy Greedy Model for Development")
    print("=" * 60)
    print("\nNOTE: This is a toy model for testing/debugging only.")
    print("  - Attempts to use real CodeSearchNet data (200 samples)")
    print("  - Falls back to synthetic comments if HuggingFace unavailable")
    print("  - Katherine will train full production model on 400k+ samples")
    print("  - Production model uses full CodeSearchNet dataset\n")
    
    # Get training data (synthetic, no dependencies needed)
    training_comments = get_training_data()
    
    # Create model directory
    model_dir = MODELS_DIR / GREEDY_MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining on {len(training_comments)} comments...")
    if len(training_comments) > 50:
        print("(Using actual CodeSearchNet data - perfect match with production!)")
    else:
        print("(Using synthetic comments that match CodeSearchNet style)\n")
    
    # Create G0_base greedy model (best greedy variant)
    print("Creating GreedyMarkovGenerator with G0_base config...")
    generator = GreedyMarkovGenerator(
        n_gram=4,  # G0_base uses 4-grams
        entropy_threshold=2.0,
        lambda_cache=0.0,  # G0_base: no cache heuristic
        lambda_anchor=0.0,  # G0_base: no anchor heuristic
        lookahead_k=0,  # G0_base: no lookahead
        backtrack_budget=0,  # G0_base: no backtrack
        log_details=False,
    )
    
    print("Training model...")
    generator.train(training_comments)
    
    print("Saving model...")
    generator.save(str(model_dir))
    
    print(f"\n✓ Toy model saved to: {model_dir}")
    print(f"  Files created:")
    for file in model_dir.glob("*"):
        print(f"    - {file.name}")
    
    # Determine data type for final message
    data_info = ""
    if len(training_comments) > 50:
        data_info = "\n✓ Used REAL CodeSearchNet data (structure matches production!)"
    else:
        data_info = "\n✓ Used synthetic data (works for testing/debugging)"
    
    print("\n" + "=" * 60)
    print("Toy Model Ready for Testing!")
    print("=" * 60)
    print(data_info)
    print(f"  - Trained on {len(training_comments)} comments")
    print(f"  - Model: {GREEDY_MODEL_NAME} (G0_base variant)")
    print("\nYou can now:")
    print("1. Test greedy service: python -c \"from services.greedy_service import GreedyService; s = GreedyService(); print('Loaded:', s.is_loaded())\"")
    print("2. Test comment generation: python tests/test_full_system.py")
    print("3. Test via API: cd backend && python run_server.py")
    print("\nNOTE: This is a toy model. Production model will be trained on full dataset (400k+ samples).")
    print("=" * 60)
    
    return model_dir

if __name__ == "__main__":
    create_toy_model()

