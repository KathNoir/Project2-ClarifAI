<<<<<<< HEAD
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import batch_compare  # type: ignore  # noqa: E402
from app.preprocess import CodeSearchNetPreprocessor  # type: ignore  # noqa: E402


DEFAULT_OUTPUT_PATH = REPO_ROOT / "reports" / "regression_results.json"


@dataclass
class RegressionConfig:
    max_new_tokens: int = 64
    beam_width: int = 4
    algorithms: Tuple[str, ...] = batch_compare.DEFAULT_ALGORITHMS


def _ensure_output_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _load_inline_or_dataset(
    inline_code: Optional[str],
    dataset_path: Optional[Path],
    *,
    language: str,
    max_samples: int,
    shuffle: bool,
) -> Iterable[Tuple[str, str]]:
    if inline_code:
        yield inline_code, language
        return

    if dataset_path:
        yield from batch_compare._load_inputs(dataset_path, None)
        return

    preprocessor = CodeSearchNetPreprocessor()
    samples = preprocessor.download_dataset(language=language, max_samples=max_samples)
    if shuffle:
        from random import shuffle as _shuffle

        _shuffle(samples)
    for item in samples[:max_samples]:
        yield item["code"], item.get("language", language)


def run_regression(
    inputs: Iterable[Tuple[str, str]],
    config: RegressionConfig,
) -> Dict[str, object]:
    algorithms = list(config.algorithms)
    winners: Dict[str, int] = {name: 0 for name in algorithms}
    metric_accumulator: Dict[str, Dict[str, List[float]]] = {
        name: {"adjusted": [], "runtime": [], "quality": [], "terminated": []}
        for name in algorithms
    }
    aggregate: List[Dict[str, object]] = []

    for idx, (code, language) in enumerate(inputs, start=1):
        outputs = batch_compare._evaluate_snippet(
            code,
            language=language,
            max_new_tokens=config.max_new_tokens,
            beam_width=config.beam_width,
        )
        outcome = batch_compare._rank_algorithms(outputs, algorithms)
        winners[outcome["preferred"]] += 1
        for name in algorithms:
            rendered = batch_compare._render_result(name, outputs[name])
            metric_accumulator[name]["adjusted"].append(rendered["adjusted_score"])  # type: ignore[arg-type]
            metric_accumulator[name]["runtime"].append(rendered["runtime_ms"])  # type: ignore[arg-type]
            metric_accumulator[name]["quality"].append(rendered["quality_score"])  # type: ignore[arg-type]
            metric_accumulator[name]["terminated"].append(
                1.0 if rendered["terminated"] else 0.0
            )  # type: ignore[arg-type]
        aggregate.append(
            {
                "index": idx,
                "language": language,
                "preferred": outcome["preferred"],
                "score_gap": outcome["score_gap"],
                "results": outcome["results"],
            }
        )

    def _avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    statistics = {
        name: {
            "avg_adjusted_score": _avg(metrics["adjusted"]),
            "avg_runtime_ms": _avg(metrics["runtime"]),
            "avg_quality_score": _avg(metrics["quality"]),
            "termination_rate": _avg(metrics["terminated"]),
        }
        for name, metrics in metric_accumulator.items()
    }

    return {
        "config": {
            "max_new_tokens": config.max_new_tokens,
            "beam_width": config.beam_width,
            "algorithms": algorithms,
        },
        "totals": {"samples": len(aggregate), "winners": winners},
        "statistics": statistics,
        "samples": aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate regression comparisons between Markov Greedy and N-gram A* Beam.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional JSONL file of code snippets. If omitted, samples will be downloaded.",
    )
    parser.add_argument(
        "--code",
        help="Inline code snippet for a quick sanity check (overrides --input and dataset download).",
    )
    parser.add_argument(
        "--language",
        default="python",
        help="Dataset language when downloading samples (default: python).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Maximum number of samples to evaluate (default: 25).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of downloaded samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination for the JSON report (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RegressionConfig(
        max_new_tokens=args.max_new_tokens,
        beam_width=args.beam_width,
        algorithms=batch_compare.DEFAULT_ALGORITHMS,
    )

    inputs = list(
        _load_inline_or_dataset(
            args.code,
            args.input,
            language=args.language,
            max_samples=args.max_samples,
            shuffle=not args.no_shuffle,
        )
    )
    if not inputs:
        raise SystemExit("No samples available for regression.")

    summary = run_regression(inputs, config)
    _ensure_output_dir(args.output)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "totals": summary["totals"],
                "statistics": summary["statistics"],
                "report_path": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

=======
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import batch_compare  # type: ignore  # noqa: E402
from app.preprocess import CodeSearchNetPreprocessor  # type: ignore  # noqa: E402


DEFAULT_OUTPUT_PATH = REPO_ROOT / "reports" / "regression_results.json"


@dataclass
class RegressionConfig:
    max_new_tokens: int = 64
    beam_width: int = 4
    algorithms: Tuple[str, ...] = batch_compare.DEFAULT_ALGORITHMS


def _ensure_output_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _load_inline_or_dataset(
    inline_code: Optional[str],
    dataset_path: Optional[Path],
    *,
    language: str,
    max_samples: int,
    shuffle: bool,
) -> Iterable[Tuple[str, str]]:
    if inline_code:
        yield inline_code, language
        return

    if dataset_path:
        yield from batch_compare._load_inputs(dataset_path, None)
        return

    preprocessor = CodeSearchNetPreprocessor()
    samples = preprocessor.download_dataset(language=language, max_samples=max_samples)
    if shuffle:
        from random import shuffle as _shuffle

        _shuffle(samples)
    for item in samples[:max_samples]:
        yield item["code"], item.get("language", language)


def run_regression(
    inputs: Iterable[Tuple[str, str]],
    config: RegressionConfig,
) -> Dict[str, object]:
    algorithms = list(config.algorithms)
    winners: Dict[str, int] = {name: 0 for name in algorithms}
    metric_accumulator: Dict[str, Dict[str, List[float]]] = {
        name: {"adjusted": [], "runtime": [], "quality": [], "terminated": []}
        for name in algorithms
    }
    aggregate: List[Dict[str, object]] = []

    for idx, (code, language) in enumerate(inputs, start=1):
        outputs = batch_compare._evaluate_snippet(
            code,
            language=language,
            max_new_tokens=config.max_new_tokens,
            beam_width=config.beam_width,
        )
        outcome = batch_compare._rank_algorithms(outputs, algorithms)
        winners[outcome["preferred"]] += 1
        for name in algorithms:
            rendered = batch_compare._render_result(name, outputs[name])
            metric_accumulator[name]["adjusted"].append(rendered["adjusted_score"])  # type: ignore[arg-type]
            metric_accumulator[name]["runtime"].append(rendered["runtime_ms"])  # type: ignore[arg-type]
            metric_accumulator[name]["quality"].append(rendered["quality_score"])  # type: ignore[arg-type]
            metric_accumulator[name]["terminated"].append(
                1.0 if rendered["terminated"] else 0.0
            )  # type: ignore[arg-type]
        aggregate.append(
            {
                "index": idx,
                "language": language,
                "preferred": outcome["preferred"],
                "score_gap": outcome["score_gap"],
                "results": outcome["results"],
            }
        )

    def _avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    statistics = {
        name: {
            "avg_adjusted_score": _avg(metrics["adjusted"]),
            "avg_runtime_ms": _avg(metrics["runtime"]),
            "avg_quality_score": _avg(metrics["quality"]),
            "termination_rate": _avg(metrics["terminated"]),
        }
        for name, metrics in metric_accumulator.items()
    }

    return {
        "config": {
            "max_new_tokens": config.max_new_tokens,
            "beam_width": config.beam_width,
            "algorithms": algorithms,
        },
        "totals": {"samples": len(aggregate), "winners": winners},
        "statistics": statistics,
        "samples": aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate regression comparisons between Markov Greedy and N-gram A* Beam.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional JSONL file of code snippets. If omitted, samples will be downloaded.",
    )
    parser.add_argument(
        "--code",
        help="Inline code snippet for a quick sanity check (overrides --input and dataset download).",
    )
    parser.add_argument(
        "--language",
        default="python",
        help="Dataset language when downloading samples (default: python).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Maximum number of samples to evaluate (default: 25).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of downloaded samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination for the JSON report (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RegressionConfig(
        max_new_tokens=args.max_new_tokens,
        beam_width=args.beam_width,
        algorithms=batch_compare.DEFAULT_ALGORITHMS,
    )

    inputs = list(
        _load_inline_or_dataset(
            args.code,
            args.input,
            language=args.language,
            max_samples=args.max_samples,
            shuffle=not args.no_shuffle,
        )
    )
    if not inputs:
        raise SystemExit("No samples available for regression.")

    summary = run_regression(inputs, config)
    _ensure_output_dir(args.output)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "totals": summary["totals"],
                "statistics": summary["statistics"],
                "report_path": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

>>>>>>> 8a0966b972509073769048bf3558c73a2ded3374
