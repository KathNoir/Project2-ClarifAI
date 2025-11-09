<<<<<<< HEAD
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from app.ngram_beam import AStarBeamGenerator
from app.preprocess import CodeSearchNetPreprocessor


def _clean_comment(text: str) -> str:
    return " ".join(text.strip().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the n-gram A* beam generator on CodeSearchNet samples."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ngram_a_star"),
        help="Directory to store the trained model artifacts.",
    )
    parser.add_argument(
        "--language",
        default="python",
        help="CodeSearchNet language subset (default: python).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Number of CodeSearchNet examples to use (default: 20000).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width for the generator during decoding (default: 5).",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=24,
        help="Target comment length used by the heuristic (default: 24 tokens).",
    )
    parser.add_argument(
        "--n-gram",
        type=int,
        default=3,
        help="Order of the n-gram model (default: trigram).",
    )
    parser.add_argument(
        "--semantic-heuristic",
        action="store_true",
        help="Enable TF-IDF semantic heuristic (requires additional memory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output

    preprocessor = CodeSearchNetPreprocessor()
    dataset = preprocessor.download_dataset(
        language=args.language, max_samples=args.max_samples
    )
    if not dataset:
        raise SystemExit("No samples retrieved from CodeSearchNet.")

    comments: List[str] = []
    code_samples: List[str] = []
    for sample in dataset:
        comment = _clean_comment(sample.get("comment", ""))
        code = sample.get("code", "")
        if comment and code:
            comments.append(comment)
            if args.semantic_heuristic:
                code_samples.append(code)

    generator = AStarBeamGenerator(
        beam_width=args.beam_width,
        alpha=0.25,
        beta=0.5,
        gamma=0.25 if args.semantic_heuristic else 0.0,
        target_length=args.target_length,
        use_semantic_heuristic=args.semantic_heuristic,
        lambda_future=0.1,
    )
    generator.train(
        comments,
        code_data=code_samples if args.semantic_heuristic else None,
        n_gram=args.n_gram,
    )
    generator.save(output_dir)

    metadata = {
        "language": args.language,
        "max_samples": args.max_samples,
        "beam_width": args.beam_width,
        "target_length": args.target_length,
        "n_gram": args.n_gram,
        "semantic_heuristic": args.semantic_heuristic,
    }
    (output_dir / "training_meta.json").write_text(json.dumps(metadata, indent=2))
    print(f"Model saved to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

=======
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from app.ngram_beam import AStarBeamGenerator
from app.preprocess import CodeSearchNetPreprocessor


def _clean_comment(text: str) -> str:
    return " ".join(text.strip().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the n-gram A* beam generator on CodeSearchNet samples."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ngram_a_star"),
        help="Directory to store the trained model artifacts.",
    )
    parser.add_argument(
        "--language",
        default="python",
        help="CodeSearchNet language subset (default: python).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Number of CodeSearchNet examples to use (default: 20000).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width for the generator during decoding (default: 5).",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=24,
        help="Target comment length used by the heuristic (default: 24 tokens).",
    )
    parser.add_argument(
        "--n-gram",
        type=int,
        default=3,
        help="Order of the n-gram model (default: trigram).",
    )
    parser.add_argument(
        "--semantic-heuristic",
        action="store_true",
        help="Enable TF-IDF semantic heuristic (requires additional memory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output

    preprocessor = CodeSearchNetPreprocessor()
    dataset = preprocessor.download_dataset(
        language=args.language, max_samples=args.max_samples
    )
    if not dataset:
        raise SystemExit("No samples retrieved from CodeSearchNet.")

    comments: List[str] = []
    code_samples: List[str] = []
    for sample in dataset:
        comment = _clean_comment(sample.get("comment", ""))
        code = sample.get("code", "")
        if comment and code:
            comments.append(comment)
            if args.semantic_heuristic:
                code_samples.append(code)

    generator = AStarBeamGenerator(
        beam_width=args.beam_width,
        alpha=0.25,
        beta=0.5,
        gamma=0.25 if args.semantic_heuristic else 0.0,
        target_length=args.target_length,
        use_semantic_heuristic=args.semantic_heuristic,
        lambda_future=0.1,
    )
    generator.train(
        comments,
        code_data=code_samples if args.semantic_heuristic else None,
        n_gram=args.n_gram,
    )
    generator.save(output_dir)

    metadata = {
        "language": args.language,
        "max_samples": args.max_samples,
        "beam_width": args.beam_width,
        "target_length": args.target_length,
        "n_gram": args.n_gram,
        "semantic_heuristic": args.semantic_heuristic,
    }
    (output_dir / "training_meta.json").write_text(json.dumps(metadata, indent=2))
    print(f"Model saved to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

>>>>>>> 8a0966b972509073769048bf3558c73a2ded3374
