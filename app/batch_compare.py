<<<<<<< HEAD
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .algorithms import (
    GenerationOutput,
    generate_markov_chain_greedy,
)
from .algorithms_ngram import generate_ngram_a_star
from .comment_utils import analyze_code


def _render_result(label: str, output: GenerationOutput) -> Dict[str, object]:
    score = output.log_probability + output.quality_score * 5.0
    return {
        "algorithm": label,
        "comment": output.comment,
        "token_count": output.token_count,
        "log_probability": output.log_probability,
        "quality_score": output.quality_score,
        "adjusted_score": score,
        "runtime_ms": output.runtime_ms,
        "num_expansions": output.num_expansions,
        "terminated": output.terminated,
        "feedback": output.feedback,
    }


DEFAULT_ALGORITHMS: Tuple[str, ...] = ("markov_greedy", "a_star_beam")


def _evaluate_snippet(
    code: str,
    *,
    language: str,
    max_new_tokens: int,
    beam_width: int,
) -> Dict[str, GenerationOutput]:
    summary = analyze_code(code, language)
    outputs: Dict[str, GenerationOutput] = {}

    def run_greedy() -> GenerationOutput:
        return generate_markov_chain_greedy(
            code,
            summary=summary,
        )

    def run_a_star() -> GenerationOutput:
        return generate_ngram_a_star(
            code,
            max_new_tokens,
            beam_width=beam_width,
            summary=summary,
        )

    dispatch = {
        "markov_greedy": run_greedy,
        "a_star_beam": run_a_star,
    }

    for name, factory in dispatch.items():
        outputs[name] = factory()

    return outputs


def _rank_algorithms(outputs: Dict[str, GenerationOutput], algorithms: Sequence[str]) -> Dict[str, object]:
    results = [_render_result(name, outputs[name]) for name in algorithms]
    ranking = sorted(results, key=lambda item: item["adjusted_score"], reverse=True)
    if len(ranking) < 2:
        score_gap = 0.0
    else:
        score_gap = ranking[0]["adjusted_score"] - ranking[1]["adjusted_score"]
    return {
        "preferred": ranking[0]["algorithm"],
        "score_gap": score_gap,
        "results": results,
    }


def _load_inputs(path: Optional[Path], inline_code: Optional[str]) -> Iterable[Tuple[str, str]]:
    if inline_code:
        yield inline_code, "python"
        return
    if path is None:
        raise ValueError("Provide --code or --input pointing to a JSONL file.")
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx}: {exc}") from exc
            if "code" not in payload:
                raise ValueError(f"Missing 'code' field on line {idx}")
            language = payload.get("language", "python")
            yield payload["code"], language


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch compare Clarify.dev generation algorithms.",
    )
    parser.add_argument(
        "--code",
        help="Inline code snippet to evaluate (overrides --input).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a JSONL file where each line has {'code': ..., 'language': ...}.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens for decoder-based generators (default: 64).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Beam width for the guided A* beam search (default: 4).",
    )
    parser.add_argument(
        "--algorithms",
        help=(
            "Comma-separated subset of algorithms to evaluate "
            f"(choices: {', '.join(DEFAULT_ALGORITHMS)}; default: all)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the aggregate JSON report.",
    )
    return parser


def _parse_algorithms(spec: Optional[str]) -> List[str]:
    if not spec:
        return list(DEFAULT_ALGORITHMS)
    requested = [part.strip().lower() for part in spec.split(",") if part.strip()]
    if not requested:
        raise ValueError("No algorithms provided.")
    invalid = [name for name in requested if name not in DEFAULT_ALGORITHMS]
    if invalid:
        raise ValueError(f"Unknown algorithms: {', '.join(invalid)}")
    # preserve input order but deduplicate while keeping default relative order
    seen: Dict[str, None] = {}
    for name in requested:
        if name not in seen:
            seen[name] = None
    if len(seen) < 2:
        raise ValueError("Provide at least two algorithms to compare.")
    return list(seen.keys())


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        inputs = list(_load_inputs(args.input, args.code))
        algorithms = _parse_algorithms(args.algorithms)
    except Exception as exc:
        parser.error(str(exc))
        return 2

    if not inputs:
        parser.error("No code snippets provided.")
        return 2

    aggregate: List[Dict[str, object]] = []
    winners: Dict[str, int] = {name: 0 for name in algorithms}
    metric_accumulator: Dict[str, Dict[str, List[float]]] = {
        name: {"adjusted": [], "runtime": [], "quality": [], "terminated": []} for name in algorithms
    }

    for idx, (code, language) in enumerate(inputs, start=1):
        outputs = _evaluate_snippet(
            code,
            language=language,
            max_new_tokens=args.max_new_tokens,
            beam_width=args.beam_width,
        )
        outcome = _rank_algorithms(outputs, algorithms)
        winners[outcome["preferred"]] += 1
        for name in algorithms:
            rendered = _render_result(name, outputs[name])
            metric_accumulator[name]["adjusted"].append(rendered["adjusted_score"])  # type: ignore[arg-type]
            metric_accumulator[name]["runtime"].append(rendered["runtime_ms"])  # type: ignore[arg-type]
            metric_accumulator[name]["quality"].append(rendered["quality_score"])  # type: ignore[arg-type]
            metric_accumulator[name]["terminated"].append(1.0 if rendered["terminated"] else 0.0)  # type: ignore[arg-type]
        aggregate.append(
            {
                "index": idx,
                "language": language,
                "preferred": outcome["preferred"],
                "score_gap": outcome["score_gap"],
                "results": outcome["results"],
            }
        )

    def _average(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    statistics = {
        name: {
            "avg_adjusted_score": _average(metrics["adjusted"]),
            "avg_runtime_ms": _average(metrics["runtime"]),
            "avg_quality_score": _average(metrics["quality"]),
            "termination_rate": _average(metrics["terminated"]),
        }
        for name, metrics in metric_accumulator.items()
    }

    summary = {
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "beam_width": args.beam_width,
            "algorithms": algorithms,
        },
        "totals": {
            "samples": len(aggregate),
            "winners": winners,
        },
        "statistics": statistics,
        "samples": aggregate,
    }

    serialized = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())

=======
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .algorithms import (
    GenerationOutput,
    generate_markov_chain_greedy,
)
from .algorithms_ngram import generate_ngram_a_star
from .comment_utils import analyze_code


def _render_result(label: str, output: GenerationOutput) -> Dict[str, object]:
    score = output.log_probability + output.quality_score * 5.0
    return {
        "algorithm": label,
        "comment": output.comment,
        "token_count": output.token_count,
        "log_probability": output.log_probability,
        "quality_score": output.quality_score,
        "adjusted_score": score,
        "runtime_ms": output.runtime_ms,
        "num_expansions": output.num_expansions,
        "terminated": output.terminated,
        "feedback": output.feedback,
    }


DEFAULT_ALGORITHMS: Tuple[str, ...] = ("markov_greedy", "a_star_beam")


def _evaluate_snippet(
    code: str,
    *,
    language: str,
    max_new_tokens: int,
    beam_width: int,
) -> Dict[str, GenerationOutput]:
    summary = analyze_code(code, language)
    outputs: Dict[str, GenerationOutput] = {}

    def run_greedy() -> GenerationOutput:
        return generate_markov_chain_greedy(
            code,
            summary=summary,
        )

    def run_a_star() -> GenerationOutput:
        return generate_ngram_a_star(
            code,
            max_new_tokens,
            beam_width=beam_width,
            summary=summary,
        )

    dispatch = {
        "markov_greedy": run_greedy,
        "a_star_beam": run_a_star,
    }

    for name, factory in dispatch.items():
        outputs[name] = factory()

    return outputs


def _rank_algorithms(outputs: Dict[str, GenerationOutput], algorithms: Sequence[str]) -> Dict[str, object]:
    results = [_render_result(name, outputs[name]) for name in algorithms]
    ranking = sorted(results, key=lambda item: item["adjusted_score"], reverse=True)
    if len(ranking) < 2:
        score_gap = 0.0
    else:
        score_gap = ranking[0]["adjusted_score"] - ranking[1]["adjusted_score"]
    return {
        "preferred": ranking[0]["algorithm"],
        "score_gap": score_gap,
        "results": results,
    }


def _load_inputs(path: Optional[Path], inline_code: Optional[str]) -> Iterable[Tuple[str, str]]:
    if inline_code:
        yield inline_code, "python"
        return
    if path is None:
        raise ValueError("Provide --code or --input pointing to a JSONL file.")
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx}: {exc}") from exc
            if "code" not in payload:
                raise ValueError(f"Missing 'code' field on line {idx}")
            language = payload.get("language", "python")
            yield payload["code"], language


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch compare Clarify.dev generation algorithms.",
    )
    parser.add_argument(
        "--code",
        help="Inline code snippet to evaluate (overrides --input).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a JSONL file where each line has {'code': ..., 'language': ...}.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens for decoder-based generators (default: 64).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Beam width for the guided A* beam search (default: 4).",
    )
    parser.add_argument(
        "--algorithms",
        help=(
            "Comma-separated subset of algorithms to evaluate "
            f"(choices: {', '.join(DEFAULT_ALGORITHMS)}; default: all)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the aggregate JSON report.",
    )
    return parser


def _parse_algorithms(spec: Optional[str]) -> List[str]:
    if not spec:
        return list(DEFAULT_ALGORITHMS)
    requested = [part.strip().lower() for part in spec.split(",") if part.strip()]
    if not requested:
        raise ValueError("No algorithms provided.")
    invalid = [name for name in requested if name not in DEFAULT_ALGORITHMS]
    if invalid:
        raise ValueError(f"Unknown algorithms: {', '.join(invalid)}")
    # preserve input order but deduplicate while keeping default relative order
    seen: Dict[str, None] = {}
    for name in requested:
        if name not in seen:
            seen[name] = None
    if len(seen) < 2:
        raise ValueError("Provide at least two algorithms to compare.")
    return list(seen.keys())


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        inputs = list(_load_inputs(args.input, args.code))
        algorithms = _parse_algorithms(args.algorithms)
    except Exception as exc:
        parser.error(str(exc))
        return 2

    if not inputs:
        parser.error("No code snippets provided.")
        return 2

    aggregate: List[Dict[str, object]] = []
    winners: Dict[str, int] = {name: 0 for name in algorithms}
    metric_accumulator: Dict[str, Dict[str, List[float]]] = {
        name: {"adjusted": [], "runtime": [], "quality": [], "terminated": []} for name in algorithms
    }

    for idx, (code, language) in enumerate(inputs, start=1):
        outputs = _evaluate_snippet(
            code,
            language=language,
            max_new_tokens=args.max_new_tokens,
            beam_width=args.beam_width,
        )
        outcome = _rank_algorithms(outputs, algorithms)
        winners[outcome["preferred"]] += 1
        for name in algorithms:
            rendered = _render_result(name, outputs[name])
            metric_accumulator[name]["adjusted"].append(rendered["adjusted_score"])  # type: ignore[arg-type]
            metric_accumulator[name]["runtime"].append(rendered["runtime_ms"])  # type: ignore[arg-type]
            metric_accumulator[name]["quality"].append(rendered["quality_score"])  # type: ignore[arg-type]
            metric_accumulator[name]["terminated"].append(1.0 if rendered["terminated"] else 0.0)  # type: ignore[arg-type]
        aggregate.append(
            {
                "index": idx,
                "language": language,
                "preferred": outcome["preferred"],
                "score_gap": outcome["score_gap"],
                "results": outcome["results"],
            }
        )

    def _average(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    statistics = {
        name: {
            "avg_adjusted_score": _average(metrics["adjusted"]),
            "avg_runtime_ms": _average(metrics["runtime"]),
            "avg_quality_score": _average(metrics["quality"]),
            "termination_rate": _average(metrics["terminated"]),
        }
        for name, metrics in metric_accumulator.items()
    }

    summary = {
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "beam_width": args.beam_width,
            "algorithms": algorithms,
        },
        "totals": {
            "samples": len(aggregate),
            "winners": winners,
        },
        "statistics": statistics,
        "samples": aggregate,
    }

    serialized = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())

>>>>>>> 8a0966b972509073769048bf3558c73a2ded3374
