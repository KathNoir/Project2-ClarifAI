from typing import List, Optional

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .preprocess import CodeSearchNetPreprocessor
from .algorithms import (
    GenerationOutput,
    generate_markov_chain_greedy,
)
from .algorithms_ngram import generate_ngram_a_star
from .comment_utils import analyze_code


MAX_CODE_CHARS = 4000

fastapi_app = FastAPI(
    title="Clarify.dev Preprocessing API",
    version="0.1.0",
    description="Lightweight FastAPI wrapper around the CodeSearchNet preprocessing helpers.",
)

preprocessor = CodeSearchNetPreprocessor()


class TokenRequest(BaseModel):
    code: str = Field(..., description="Source code snippet to analyze.")


class GenerateRequest(BaseModel):
    code: str = Field(
        ...,
        min_length=8,
        max_length=MAX_CODE_CHARS,
        description="Code snippet to annotate.",
    )
    language: str = Field("python", max_length=32)
    max_new_tokens: int = Field(64, ge=16, le=256)
    beam_width: int = Field(
        4,
        ge=1,
        le=12,
        description="Beam width used by the n-gram A* generator.",
    )


class AlgorithmResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    comment: str
    token_count: int
    log_probability: float
    average_log_probability: float
    runtime_ms: float
    num_expansions: int
    terminated: bool
    token_sequence: List[str]
    quality_score: float
    feedback: List[str]


class ComparisonSummary(BaseModel):
    preferred: str
    score_gap: float
    notes: Optional[str] = None


class GenerateResponse(BaseModel):
    markov_greedy: AlgorithmResult
    a_star_beam: AlgorithmResult
    comparison: ComparisonSummary


class DownloadRequest(BaseModel):
    language: str = Field("python", description="CodeSearchNet language subset.")
    max_samples: Optional[int] = Field(
        default=200,
        ge=1,
        le=5000,
        description="Maximum number of samples to fetch for quick demos.",
    )


@fastapi_app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "message": "Clarify.dev preprocessing helpers are ready.",
    }


@fastapi_app.post("/tokens", response_model=List[str])
def extract_tokens(payload: TokenRequest) -> List[str]:
    tokens = preprocessor.extract_code_tokens(payload.code)
    if not tokens:
        raise HTTPException(status_code=422, detail="No tokens extracted from code snippet.")
    return tokens


@fastapi_app.post("/sample-dataset")
def sample_dataset(payload: DownloadRequest) -> dict:
    data = preprocessor.download_dataset(language=payload.language, max_samples=payload.max_samples)
    if not data:
        raise HTTPException(status_code=500, detail="Failed to download dataset. Check logs.")

    preview = data[0] if data else {}
    return {
        "total_samples": len(data),
        "preview": {
            "function_name": preview.get("func_name"),
            "comment": preview.get("comment"),
            "code_snippet": preview.get("code", "")[:200],
            "split": preview.get("split"),
        },
    }


@fastapi_app.post("/split")
def split_dataset(items: List[dict]) -> dict:
    if not items:
        raise HTTPException(status_code=400, detail="Provide at least one item.")

    train, dev, test = preprocessor.create_train_dev_test_split(items)
    return {
        "train_count": len(train),
        "dev_count": len(dev),
        "test_count": len(test),
    }


def _convert_output(output: GenerationOutput) -> AlgorithmResult:
    return AlgorithmResult.model_validate(output)


def _run_generation(payload: GenerateRequest) -> GenerateResponse:
    summary = analyze_code(payload.code, payload.language)
    greedy_output = generate_markov_chain_greedy(
        payload.code,
        summary=summary,
    )
    a_star_output = generate_ngram_a_star(
        payload.code,
        payload.max_new_tokens,
        summary=summary,
        beam_width=payload.beam_width,
    )

    greedy_result = _convert_output(greedy_output)
    a_star_result = _convert_output(a_star_output)

    adjusted_scores = {
        "markov_greedy": greedy_result.log_probability + greedy_result.quality_score * 5.0,
        "a_star_beam": a_star_result.log_probability + a_star_result.quality_score * 5.0,
    }
    ranking = sorted(adjusted_scores.items(), key=lambda item: item[1], reverse=True)
    preferred = ranking[0][0]
    runner_up_score = ranking[1][1] if len(ranking) > 1 else ranking[0][1]
    gap = ranking[0][1] - runner_up_score

    notes: List[str] = []
    if not greedy_result.terminated:
        notes.append("Markov chain generator exhausted states before explicit closing.")
    if not a_star_result.terminated:
        notes.append("N-gram A* beam search abandoned before reaching EOS.")
    if payload.max_new_tokens >= 200:
        notes.append("Long outputs may be truncated.")

    return GenerateResponse(
        markov_greedy=greedy_result,
        a_star_beam=a_star_result,
        comparison=ComparisonSummary(
            preferred=preferred,
            score_gap=gap,
            notes="; ".join(notes) if notes else None,
        ),
    )


@fastapi_app.post("/generate", response_model=GenerateResponse)
def generate_comments(payload: GenerateRequest) -> GenerateResponse:
    try:
        return _run_generation(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


def _format_metrics(result: AlgorithmResult) -> dict:
    return {
        "token_count": result.token_count,
        "log_probability": round(result.log_probability, 4),
        "avg_log_probability": round(result.average_log_probability, 4),
        "runtime_ms": round(result.runtime_ms, 2),
        "num_expansions": result.num_expansions,
        "terminated": result.terminated,
        "quality_score": round(result.quality_score, 3),
    }


def generate_ui(
    code: str,
    language: str,
    max_new_tokens: int,
    beam_width: int,
):
    cleaned = (code or "").strip()
    if not cleaned:
        raise gr.Error("Please paste a code snippet to analyze.")
    if len(cleaned) > MAX_CODE_CHARS:
        raise gr.Error(f"Code snippet exceeds {MAX_CODE_CHARS} characters.")

    payload = GenerateRequest(
        code=cleaned,
        language=language,
        max_new_tokens=int(max_new_tokens),
        beam_width=int(beam_width),
    )
    response = _run_generation(payload)
    adjusted_scores = {
        "markov_greedy": response.markov_greedy.log_probability + response.markov_greedy.quality_score * 5.0,
        "a_star_beam": response.a_star_beam.log_probability + response.a_star_beam.quality_score * 5.0,
    }
    ranking = sorted(adjusted_scores.items(), key=lambda item: item[1], reverse=True)
    summary_lines = [
        f"**Preferred:** {response.comparison.preferred.upper()}",
        f"Δ adjusted score (winner - runner-up): {response.comparison.score_gap:.4f}",
    ]
    if response.comparison.notes:
        summary_lines.append(f"_Notes_: {response.comparison.notes}")
    summary_lines.append("**Adjusted Scores (log prob + 5×quality)**")
    label_map = {
        "markov_greedy": "Markov Greedy",
        "a_star_beam": "N-gram A* Beam",
    }
    result_map = {
        "markov_greedy": response.markov_greedy,
        "a_star_beam": response.a_star_beam,
    }
    for algo_key, score in ranking:
        result = result_map[algo_key]
        summary_lines.append(
            f"- {label_map[algo_key]}: {score:.4f} (quality {result.quality_score:.3f}, terminated={result.terminated})"
        )

    return (
        response.markov_greedy.comment,
        _format_metrics(response.markov_greedy),
        response.a_star_beam.comment,
        _format_metrics(response.a_star_beam),
        "\n".join(
            summary_lines
            + [
                f"- Markov Greedy feedback: {', '.join(response.markov_greedy.feedback) or '✅ Balanced'}",
                f"- N-gram A* Beam feedback: {', '.join(response.a_star_beam.feedback) or '✅ Balanced'}",
            ]
        ),
    )


with gr.Blocks(title="Clarify.dev API") as demo:
    gr.Markdown(
        """
        # Clarify.dev
        Compare a Markov-chain greedy generator with a coverage-aware A* beam search for actionable documentation comments.
        Use the panel below or hit the REST endpoints under `/docs`.
        """
    )
    with gr.Row():
        code_box = gr.Textbox(
            label="Code snippet",
            lines=18,
            placeholder="Paste up to 4,000 characters of code…",
        )
    with gr.Row():
        language_box = gr.Dropdown(
            ["python", "javascript", "java", "go", "c++", "c#", "ruby"],
            value="python",
            label="Language (metadata only)",
        )
        max_tokens_slider = gr.Slider(
            minimum=16,
            maximum=160,
            value=64,
            step=8,
            label="Max new comment tokens",
        )
        beam_slider = gr.Slider(
            minimum=1,
            maximum=12,
            value=4,
            step=1,
            label="A* beam width",
        )
    run_button = gr.Button("Generate Comments")
    with gr.Row():
        greedy_comment = gr.Textbox(
            label="Markov Greedy Comment",
            lines=8,
        )
        greedy_metrics = gr.JSON(label="Markov Greedy Metrics")
    with gr.Row():
        astar_comment = gr.Textbox(
            label="N-gram A* Beam Comment",
            lines=8,
        )
        astar_metrics = gr.JSON(label="N-gram A* Beam Metrics")
    summary_box = gr.Markdown()

    run_button.click(
        fn=generate_ui,
        inputs=[
            code_box,
            language_box,
            max_tokens_slider,
            beam_slider,
        ],
        outputs=[
            greedy_comment,
            greedy_metrics,
            astar_comment,
            astar_metrics,
            summary_box,
        ],
    )

app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)

