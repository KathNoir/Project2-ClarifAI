from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .comment_utils import CodeSummary, analyze_code, score_comment
from .ngram_beam import AStarBeamGenerator
from .preprocess import CodeSearchNetPreprocessor
from .algorithms import GenerationOutput

_MODEL_DIR = Path(os.getenv("CLARIFY_NGRAM_MODEL_DIR", "models/ngram_a_star"))
_TOKEN_EXTRACTOR = CodeSearchNetPreprocessor()


@lru_cache(maxsize=1)
def _load_ngram_generator() -> AStarBeamGenerator:
    if not _MODEL_DIR.exists():
        raise RuntimeError(
            f"N-gram model directory not found at {_MODEL_DIR.resolve()}. "
            "Train the model first with `python -m scripts.train_ngram_model`."
        )
    return AStarBeamGenerator.load(_MODEL_DIR)


def generate_ngram_a_star(
    code: str,
    max_new_tokens: int,
    *,
    summary: Optional[CodeSummary] = None,
    beam_width: Optional[int] = None,
) -> GenerationOutput:
    summary = summary or analyze_code(code)
    generator = _load_ngram_generator()
    code_tokens = _TOKEN_EXTRACTOR.extract_code_tokens(code, limit=20)

    original_beam_width = generator.beam_width
    if beam_width and beam_width > 0:
        generator.beam_width = beam_width

    start = time.perf_counter()
    comment, metadata = generator.generate(
        code_tokens,
        code_text=code,
        max_length=max_new_tokens,
        return_metadata=True,
    )
    runtime_ms = (time.perf_counter() - start) * 1000.0

    if beam_width and beam_width > 0:
        generator.beam_width = original_beam_width

    comment = comment.strip()
    comment_lower = comment.lower()
    anchor_tokens = [tok for tok in summary.anchor_tokens if tok]
    missing: List[str] = []
    if summary.function_name:
        fn = summary.function_name.lower()
        if fn and fn not in comment_lower:
            missing.append(summary.function_name)
    for token in anchor_tokens:
        if token.lower() not in comment_lower and token not in missing:
            missing.append(token)
    if missing:
        focus = f"`{missing[0]}`"
        addition = f" Highlights {focus} to keep the summary grounded."
        if comment:
            if not comment.endswith("."):
                comment += "."
            comment += addition
        else:
            comment = addition.strip()
        comment_lower = comment.lower()

    token_sequence = comment.split()
    token_count = len(token_sequence)
    log_probability = float(metadata.get("log_prob", 0.0))
    average_log_probability = float(metadata.get("avg_log_prob", log_probability))
    quality, feedback = score_comment(comment, summary)

    return GenerationOutput(
        comment=comment,
        token_count=token_count,
        log_probability=log_probability,
        average_log_probability=average_log_probability,
        runtime_ms=runtime_ms,
        num_expansions=int(metadata.get("expansions", 0)),
        terminated=bool(metadata.get("terminated", False)),
        token_sequence=token_sequence,
        quality_score=quality,
        feedback=feedback,
    )

