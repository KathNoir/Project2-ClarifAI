from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import heapq

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .comment_utils import CodeSummary, analyze_code, build_prompt, score_comment


DEFAULT_MODEL_NAME = os.getenv("CLARIFY_MODEL_NAME", "Salesforce/codet5-small")
MAX_ENCODER_TOKENS = int(os.getenv("CLARIFY_MAX_ENCODER_TOKENS", "512"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GenerationOutput:
    comment: str
    token_count: int
    log_probability: float
    average_log_probability: float
    runtime_ms: float
    num_expansions: int
    terminated: bool
    token_sequence: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    feedback: List[str] = field(default_factory=list)


@dataclass(order=True)
class _SearchNode:
    priority: float
    log_prob: float = field(compare=False)
    tokens: Tuple[int, ...] = field(compare=False)
    upper_bound: float = field(compare=False)
    terminated: bool = field(compare=False)


@dataclass
class _MarkovStep:
    text: Optional[str]
    options: Sequence[Tuple[str, float]]


class ModelNotReadyError(RuntimeError):
    """Raised when the model assets fail to load."""


@lru_cache(maxsize=1)
def _load_assets():
    try:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL_NAME)
    except Exception as exc:  # pragma: no cover - initialization failure should surface quickly
        raise ModelNotReadyError(str(exc)) from exc

    model.to(DEVICE)
    model.eval()
    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id
    return tokenizer, model, decoder_start


@dataclass
class GenerationContext:
    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM
    decoder_start_token_id: int
    encoder_outputs: object
    prompt: str


def build_context(code: str, language: str = "python") -> Tuple[GenerationContext, CodeSummary]:
    summary = analyze_code(code, language)
    prompt = build_prompt(code, summary)
    tokenizer, model, decoder_start_token_id = _load_assets()
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_ENCODER_TOKENS,
    )
    encoded = {name: tensor.to(DEVICE) for name, tensor in encoded.items()}
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(**encoded)
    context = GenerationContext(
        tokenizer=tokenizer,
        model=model,
        decoder_start_token_id=decoder_start_token_id,
        encoder_outputs=encoder_outputs,
        prompt=prompt,
    )
    return context, summary


def _decode_tokens(tokenizer, token_ids: Iterable[int]) -> Tuple[str, List[str], int]:
    special_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    visible_ids = [tid for tid in token_ids if tid not in special_ids]
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    tokens = tokenizer.convert_ids_to_tokens(visible_ids)
    return text, tokens, len(visible_ids)


def generate_greedy(
    code: str,
    max_new_tokens: int,
    *,
    repetition_penalty: float = 1.0,
    context: Optional[GenerationContext] = None,
    summary: Optional[CodeSummary] = None,
) -> GenerationOutput:
    if context is None or summary is None:
        ctx, summary = build_context(code)
    else:
        ctx = context
    summary = summary or analyze_code(code)
    tokenizer = ctx.tokenizer
    model = ctx.model
    decoder_start_token_id = ctx.decoder_start_token_id
    encoder_outputs = ctx.encoder_outputs

    start = time.perf_counter()
    decoder_input_ids = torch.tensor([[decoder_start_token_id]], device=DEVICE)

    generated_ids: List[int] = []
    log_prob_total = 0.0
    expansions = 0

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            logits = _apply_repetition_penalty_to_logits(
                logits, generated_ids, repetition_penalty
            )
            log_probs = F.log_softmax(logits, dim=-1)
            next_token_id = torch.argmax(log_probs, dim=-1)
            log_prob_value = log_probs[0, next_token_id].item()

            token_id = next_token_id.item()
            generated_ids.append(token_id)
            log_prob_total += log_prob_value
            expansions += 1

            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token_id.unsqueeze(0)], dim=-1
            )

            if token_id == tokenizer.eos_token_id:
                break

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    text, tokens, token_count = _decode_tokens(tokenizer, generated_ids)
    if not text:
        text = "Summarize the purpose and behavior of this code."
    avg_log_prob = (
        log_prob_total / token_count if token_count else float("-inf")
    )
    terminated = bool(generated_ids and generated_ids[-1] == tokenizer.eos_token_id)
    quality, feedback = score_comment(text, summary)

    return GenerationOutput(
        comment=text,
        token_count=token_count,
        log_probability=log_prob_total,
        average_log_probability=avg_log_prob,
        runtime_ms=elapsed_ms,
        num_expansions=expansions,
        terminated=terminated,
        token_sequence=tokens,
        quality_score=quality,
        feedback=feedback,
    )


def _upper_bound(log_prob: float, steps: int, max_steps: int) -> float:
    if steps == 0:
        optimistic = -0.1
    else:
        avg = log_prob / steps
        optimistic = max(avg, -0.1)
    remaining = max(max_steps - steps, 0)
    return log_prob + optimistic * remaining


def _apply_repetition_penalty_to_logits(
    logits: torch.Tensor,
    generated_ids: Sequence[int],
    penalty: float,
) -> torch.Tensor:
    if penalty <= 1.0 or not generated_ids:
        return logits
    unique_ids = {int(token_id) for token_id in generated_ids}
    if not unique_ids:
        return logits
    adjusted = logits.clone()
    for token_id in unique_ids:
        token_slice = adjusted[..., token_id]
        adjusted[..., token_id] = torch.where(
            token_slice < 0,
            token_slice * penalty,
            token_slice / penalty,
        )
    return adjusted


def _clean_anchor_tokens(summary: CodeSummary, *, min_length: int = 3) -> List[str]:
    return sorted(
        {tok for tok in summary.anchor_tokens if len(tok) >= min_length},
        key=len,
    )


_SECTION_BY_STATE = {
    "overview": "overview",
    "parameters": "parameters",
    "returns": "returns",
    "calls": "calls",
    "complexity": "complexity",
    "closing": "closing",
}


def _state_has_content(state: str, summary: CodeSummary) -> bool:
    if state == "parameters":
        return bool(summary.parameters)
    if state == "returns":
        return bool(summary.return_keywords)
    if state == "calls":
        return bool(summary.called_functions)
    if state == "complexity":
        return summary.complexity in {"moderate", "complex"}
    return True


def _select_enhanced_state(
    options: Sequence[Tuple[str, float]],
    *,
    visited: Set[str],
    covered_sections: Set[str],
    summary: CodeSummary,
    min_sections: int,
    current_sections: int,
) -> Optional[str]:
    best_state: Optional[str] = None
    best_score = float("-inf")

    anchor_sections_remaining = max(len(_SECTION_BY_STATE) - len(covered_sections), 1)
    for state, weight in options:
        if state == "__END__":
            if current_sections < min_sections:
                continue
        if state != "__END__" and not _state_has_content(state, summary):
            continue

        penalty = 0.4 if state in visited else 0.0
        section = _SECTION_BY_STATE.get(state)
        coverage_bonus = 0.0
        if section and section not in covered_sections:
            coverage_bonus = 0.6 + (0.2 / anchor_sections_remaining)
        score = weight + coverage_bonus - penalty

        if score > best_score:
            best_score = score
            best_state = state
    return best_state


def _run_markov_chain(
    summary: CodeSummary,
    *,
    max_steps: int = 7,
    min_sections: int = 4,
) -> Tuple[str, int, bool]:
    state = "overview"
    visited: Set[str] = set()
    covered_sections: Set[str] = set()
    phrases: List[str] = []
    transitions = 0
    terminated = False

    while transitions < max_steps:
        generator = _MARKOV_STATE_MAP.get(state)
        if generator is None:
            break

        result = generator(summary)
        if result.text:
            phrases.append(result.text)
            section = _SECTION_BY_STATE.get(state)
            if section:
                covered_sections.add(section)

        if state != "__START__":
            visited.add(state)

        next_state = _select_enhanced_state(
            result.options,
            visited=visited,
            covered_sections=covered_sections,
            summary=summary,
            min_sections=min_sections,
            current_sections=len(covered_sections),
        )
        if next_state is None:
            break
        if next_state == "__END__":
            terminated = True
            break

        state = next_state
        transitions += 1

    comment = " ".join(phrases).strip()
    return comment, transitions, terminated


def generate_markov_chain_greedy(
    code: str,
    *,
    summary: Optional[CodeSummary] = None,
    min_sections: int = 4,
    max_steps: int = 8,
) -> GenerationOutput:
    summary = summary or analyze_code(code)
    start = time.perf_counter()
    comment, transitions, terminated = _run_markov_chain(
        summary, max_steps=max_steps, min_sections=min_sections
    )
    if not comment:
        comment = "This helper documents the intent of the code by outlining its key behaviors."
        terminated = False
    runtime_ms = (time.perf_counter() - start) * 1000.0
    tokens = comment.split()
    quality, feedback = score_comment(comment, summary)

    return GenerationOutput(
        comment=comment,
        token_count=len(tokens),
        log_probability=0.0,
        average_log_probability=0.0,
        runtime_ms=runtime_ms,
        num_expansions=transitions,
        terminated=terminated,
        token_sequence=tokens,
        quality_score=quality,
        feedback=feedback,
    )


def generate_a_star(
    code: str,
    max_new_tokens: int,
    *,
    branching_factor: int = 4,
    max_expansions: int = 256,
    repetition_penalty: float = 1.0,
    context: Optional[GenerationContext] = None,
    summary: Optional[CodeSummary] = None,
) -> GenerationOutput:
    if context is None or summary is None:
        ctx, summary = build_context(code)
    else:
        ctx = context
    summary = summary or analyze_code(code)
    tokenizer = ctx.tokenizer
    model = ctx.model
    decoder_start_token_id = ctx.decoder_start_token_id
    encoder_outputs = ctx.encoder_outputs

    start = time.perf_counter()
    frontier: List[_SearchNode] = []
    root_upper = _upper_bound(0.0, 0, max_new_tokens)
    heapq.heappush(
        frontier,
        _SearchNode(priority=-root_upper, log_prob=0.0, tokens=tuple(), upper_bound=root_upper, terminated=False),
    )

    best_node: Optional[_SearchNode] = None
    best_terminated: Optional[_SearchNode] = None
    expansions = 0

    with torch.no_grad():
        while frontier and expansions < max_expansions:
            current = heapq.heappop(frontier)

            if current.terminated or len(current.tokens) >= max_new_tokens:
                if best_node is None or current.log_prob > best_node.log_prob:
                    best_node = current
                if current.terminated and (
                    best_terminated is None or current.log_prob > best_terminated.log_prob
                ):
                    best_terminated = current
                if not frontier:
                    break
                if best_terminated and best_terminated.log_prob >= frontier[0].upper_bound:
                    break
                continue

            decoder_tokens = [decoder_start_token_id, *current.tokens]
            decoder_input_ids = torch.tensor([decoder_tokens], device=DEVICE)

            outputs = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            logits = _apply_repetition_penalty_to_logits(
                logits, current.tokens, repetition_penalty
            )
            log_probs = F.log_softmax(logits, dim=-1)[0]
            top_log_probs, top_indices = torch.topk(
                log_probs, k=min(branching_factor, log_probs.shape[-1])
            )

            for log_prob_value, token_idx in zip(top_log_probs, top_indices):
                token_id = int(token_idx.item())
                new_tokens = current.tokens + (token_id,)
                new_log_prob = current.log_prob + float(log_prob_value.item())
                terminated = token_id == tokenizer.eos_token_id
                new_steps = len(new_tokens)
                upper = _upper_bound(new_log_prob, new_steps, max_new_tokens)
                heapq.heappush(
                    frontier,
                    _SearchNode(
                        priority=-upper,
                        log_prob=new_log_prob,
                        tokens=new_tokens,
                        upper_bound=upper,
                        terminated=terminated,
                    ),
                )

            expansions += 1

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if best_terminated is None and frontier:
        terminated_candidates = [node for node in frontier if node.terminated]
        if terminated_candidates:
            best_terminated = max(terminated_candidates, key=lambda n: n.log_prob)
    if best_terminated is not None:
        best_node = best_terminated
    elif best_node is None and frontier:
        candidate = max(frontier, key=lambda n: n.log_prob)
        if candidate.log_prob != float("-inf"):
            best_node = candidate
    if best_node is None:
        best_node = _SearchNode(priority=0.0, log_prob=float("-inf"), tokens=tuple(), upper_bound=float("-inf"), terminated=False)

    text, tokens, token_count = _decode_tokens(tokenizer, best_node.tokens)
    avg_log_prob = (
        best_node.log_prob / token_count if token_count else float("-inf")
    )
    terminated = bool(best_node.tokens and best_node.tokens[-1] == tokenizer.eos_token_id)
    quality, feedback = score_comment(text, summary)

    return GenerationOutput(
        comment=text,
        token_count=token_count,
        log_probability=best_node.log_prob,
        average_log_probability=avg_log_prob,
        runtime_ms=elapsed_ms,
        num_expansions=expansions,
        terminated=terminated,
        token_sequence=tokens,
        quality_score=quality,
        feedback=feedback,
    )


def generate_guided_a_star_beam(
    code: str,
    max_new_tokens: int,
    *,
    branching_factor: int = 4,
    beam_width: int = 4,
    max_expansions: int = 384,
    repetition_penalty: float = 1.1,
    context: Optional[GenerationContext] = None,
    summary: Optional[CodeSummary] = None,
    **_: object,
) -> GenerationOutput:
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if context is None or summary is None:
        ctx, summary = build_context(code)
    else:
        ctx = context
    summary = summary or analyze_code(code)

    tokenizer = ctx.tokenizer
    model = ctx.model
    decoder_start_token_id = ctx.decoder_start_token_id
    encoded = tokenizer(
        ctx.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_ENCODER_TOKENS,
    )
    encoded = {name: tensor.to(DEVICE) for name, tensor in encoded.items()}
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    start_time = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            decoder_start_token_id=decoder_start_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=max(beam_width, 2),
            early_stopping=True,
            repetition_penalty=repetition_penalty,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad_token_id,
            num_return_sequences=1,
        )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    sequence = generated.sequences[0].tolist()
    generated_ids = sequence[1:]
    text, tokens, token_count = _decode_tokens(tokenizer, generated_ids)
    if not text:
        text = "Summarize the routine's purpose, inputs, and edge cases."
        tokens = text.split()
        token_count = len(tokens)

    log_prob_total = (
        generated.sequences_scores[0].item() if generated.sequences_scores is not None else 0.0
    )
    avg_log_prob = (
        log_prob_total / token_count if token_count else float("-inf")
    )
    terminated = bool(generated_ids and generated_ids[-1] == tokenizer.eos_token_id)
    expansions = min(len(generated.scores) * max(beam_width, branching_factor), max_expansions)

    quality, feedback = score_comment(text, summary)

    return GenerationOutput(
        comment=text,
        token_count=token_count,
        log_probability=log_prob_total,
        average_log_probability=avg_log_prob,
        runtime_ms=elapsed_ms,
        num_expansions=expansions,
        terminated=terminated,
        token_sequence=tokens,
        quality_score=quality,
        feedback=feedback,
    )




def _select_weighted_state(
    options: Sequence[Tuple[str, float]],
    *,
    visited: Set[str],
) -> Optional[str]:
    best_state: Optional[str] = None
    best_score = float("-inf")
    for state, weight in options:
        if weight <= 0:
            continue
        adjusted = weight - (0.35 if state in visited else 0.0)
        if adjusted > best_score:
            best_state = state
            best_score = adjusted
    return best_state


def _format_sequence(items: Sequence[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _pick_anchor_tokens(summary: CodeSummary, limit: int = 3) -> List[str]:
    anchors = sorted(summary.anchor_tokens)
    if summary.function_name and summary.function_name.lower() in summary.anchor_tokens:
        anchors.insert(0, summary.function_name.lower())
    unique = []
    seen: Set[str] = set()
    for token in anchors:
        if token and token not in seen:
            unique.append(token)
            seen.add(token)
        if len(unique) >= limit:
            break
    return unique


def _state_overview(summary: CodeSummary) -> _MarkovStep:
    subject_options: List[Tuple[str, float]] = [
        ("This helper", 0.9),
        ("This function", 1.0),
        ("The routine", 0.6),
    ]
    if summary.function_name:
        subject_options.append((f"The `{summary.function_name}` routine", 1.25))

    subject = max(subject_options, key=lambda item: item[1])[0]
    verbs_by_complexity = {
        "simple": ["summarizes", "wraps", "prepares"],
        "moderate": ["coordinates", "validates", "routes"],
        "complex": ["orchestrates", "synthesizes", "manages"],
    }
    verb = verbs_by_complexity.get(summary.complexity, ["handles"])[0]

    anchors = _pick_anchor_tokens(summary, limit=2)
    anchor_phrase = ""
    if anchors:
        anchor_phrase = f" around {_format_sequence([f'`{tok}`' for tok in anchors])}"

    if summary.called_functions:
        call_phrase = f" by leveraging `{summary.called_functions[0]}`"
    else:
        call_phrase = ""

    overview = f"{subject} {verb}{anchor_phrase}{call_phrase}."
    options: List[Tuple[str, float]] = []
    if summary.parameters:
        options.append(("parameters", 1.1))
    if summary.return_keywords:
        options.append(("returns", 1.0))
    if summary.called_functions:
        options.append(("calls", 0.9))
    if summary.complexity in {"moderate", "complex"} or summary.num_lines > 20:
        options.append(("complexity", 0.7))
    options.append(("closing", 0.5))
    return _MarkovStep(overview, options)


def _state_parameters(summary: CodeSummary) -> _MarkovStep:
    params = summary.parameters[:3]
    if not params:
        return _MarkovStep(None, [("returns", 0.9), ("closing", 0.4)])
    formatted = _format_sequence([f"`{name}`" for name in params])
    detail = (
        f"It validates {formatted} to guard against malformed inputs."
        if summary.complexity != "simple"
        else f"It accepts {formatted} and applies consistent formatting."
    )
    return _MarkovStep(detail, [("returns", 1.0), ("calls", 0.6), ("closing", 0.5)])


def _state_returns(summary: CodeSummary) -> _MarkovStep:
    returns = summary.return_keywords[:2]
    if not returns:
        return _MarkovStep(
            "It produces a structured comment that highlights the important behaviors.",
            [("closing", 0.8)],
        )
    formatted = _format_sequence([f"`{tok}`" for tok in returns])
    detail = f"It returns {formatted} so downstream callers can reuse the derived context."
    return _MarkovStep(detail, [("closing", 0.9), ("complexity", 0.5)])


def _state_calls(summary: CodeSummary) -> _MarkovStep:
    calls = summary.called_functions[:2]
    if not calls:
        return _MarkovStep(None, [("complexity", 0.6), ("closing", 0.7)])
    formatted = _format_sequence([f"`{call}`" for call in calls])
    detail = f"It coordinates calls to {formatted} so side effects stay localized."
    return _MarkovStep(detail, [("closing", 0.8), ("complexity", 0.6)])


def _state_complexity(summary: CodeSummary) -> _MarkovStep:
    if summary.complexity == "simple":
        text = "The control flow stays straightforward, keeping edge cases predictable."
    elif summary.complexity == "moderate":
        text = "Branches and loops are balanced to address the primary edge cases without bloating the path."
    else:
        text = "It juggles multiple branches, so ensure new callers cover alternative paths with tests."
    return _MarkovStep(text, [("closing", 0.9)])


def _state_closing(summary: CodeSummary) -> _MarkovStep:
    anchor_tokens = _pick_anchor_tokens(summary, limit=3)
    anchors = (
        f"{_format_sequence([f'`{tok}`' for tok in anchor_tokens])} "
        if anchor_tokens
        else ""
    )
    closing = (
        f"Document scenarios where {anchors}evolve to keep the comment actionable."
        if anchor_tokens
        else "Document the surrounding edge cases so the comment stays actionable."
    )
    return _MarkovStep(closing, [("__END__", 1.0)])


_MARKOV_STATE_MAP: Dict[str, Callable[[CodeSummary], _MarkovStep]] = {
    "__START__": lambda summary: _MarkovStep(None, [("overview", 1.0)]),
    "overview": _state_overview,
    "parameters": _state_parameters,
    "returns": _state_returns,
    "calls": _state_calls,
    "complexity": _state_complexity,
    "closing": _state_closing,
}


def _run_markov(summary: CodeSummary, max_steps: int = 6) -> Tuple[str, int, bool]:
    state = "__START__"
    visited: Set[str] = set()
    phrases: List[str] = []
    steps = 0
    terminated = False

    while steps < max_steps:
        generator = _MARKOV_STATE_MAP.get(state)
        if generator is None:
            break

        step_result = generator(summary)
        if step_result.text:
            phrases.append(step_result.text)

        if state != "__START__":
            visited.add(state)

        next_state = _select_weighted_state(step_result.options, visited=visited)
        if next_state is None:
            break
        if next_state == "__END__":
            terminated = True
            break

        state = next_state
        steps += 1

    comment = " ".join(phrases).strip()
    return comment, steps, terminated


def generate_markov_greedy(
    code: str,
    *,
    summary: Optional[CodeSummary] = None,
) -> GenerationOutput:
    summary = summary or analyze_code(code)
    start_time = time.perf_counter()
    comment, transitions, terminated = _run_markov(summary)
    if not comment:
        comment = "This helper documents the intent of the code by outlining its key behaviors."
        terminated = False
    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    tokens = comment.split()
    quality, feedback = score_comment(comment, summary)

    return GenerationOutput(
        comment=comment,
        token_count=len(tokens),
        log_probability=0.0,
        average_log_probability=0.0,
        runtime_ms=runtime_ms,
        num_expansions=transitions,
        terminated=terminated,
        token_sequence=tokens,
        quality_score=quality,
        feedback=feedback,
    )

