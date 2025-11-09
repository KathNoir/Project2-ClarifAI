from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass
class CodeSummary:
    language: str
    function_name: Optional[str]
    parameters: List[str]
    return_keywords: List[str]
    raises: List[str]
    called_functions: List[str]
    complexity: str
    docstring_present: bool
    num_lines: int
    anchor_tokens: Set[str] = field(default_factory=set)


KEYWORD_VERBS = {
    "return",
    "compute",
    "calculate",
    "build",
    "fetch",
    "retrieve",
    "create",
    "update",
    "delete",
    "validate",
    "normalize",
    "parse",
    "format",
    "check",
    "ensure",
    "generate",
}


def analyze_code(code: str, language: str = "python") -> CodeSummary:
    if language.lower() != "python":
        return _fallback_summary(code, language)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _fallback_summary(code, language)

    function_name = None
    parameters: List[str] = []
    return_keywords: List[str] = []
    raises: List[str] = []
    called_functions: Set[str] = set()
    num_lines = code.count("\n") + 1
    docstring_present = False
    loops = 0
    conditionals = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and function_name is None:
            function_name = node.name
            parameters = [arg.arg for arg in node.args.args if arg.arg != "self"]
            docstring_present = ast.get_docstring(node) is not None
        elif isinstance(node, ast.Return):
            if isinstance(node.value, ast.Name):
                return_keywords.append(node.value.id)
            elif isinstance(node.value, ast.Attribute):
                return_keywords.append(node.value.attr)
        elif isinstance(node, ast.Raise):
            if isinstance(node.exc, ast.Name):
                raises.append(node.exc.id)
            elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                raises.append(node.exc.func.id)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called_functions.add(func.id)
            elif isinstance(func, ast.Attribute):
                called_functions.add(func.attr)
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            loops += 1
        elif isinstance(node, (ast.If, ast.Match)):
            conditionals += 1

    complexity_level = "simple"
    if loops + conditionals > 3:
        complexity_level = "complex"
    elif loops + conditionals > 1:
        complexity_level = "moderate"

    anchor_tokens = {
        *(parameters),
        *(return_keywords),
        *(raises),
        *(called_functions),
    }
    if function_name:
        anchor_tokens.add(function_name)

    return CodeSummary(
        language=language,
        function_name=function_name,
        parameters=parameters,
        return_keywords=return_keywords,
        raises=raises,
        called_functions=list(called_functions),
        complexity=complexity_level,
        docstring_present=docstring_present,
        num_lines=num_lines,
        anchor_tokens={tok.lower() for tok in anchor_tokens if tok},
    )


def _fallback_summary(code: str, language: str) -> CodeSummary:
    identifiers = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
    tokens = set(tok.lower() for tok in identifiers if len(tok) > 1)
    function_name = next((tok for tok in identifiers if tok.islower()), None)

    return CodeSummary(
        language=language,
        function_name=function_name,
        parameters=[],
        return_keywords=[],
        raises=[],
        called_functions=[],
        complexity="unknown",
        docstring_present=False,
        num_lines=code.count("\n") + 1,
        anchor_tokens=tokens,
    )


def build_prompt(code: str, summary: CodeSummary) -> str:
    header_parts = [
        "You are Clarify.dev, an expert engineer who writes precise, helpful code comments.",
        f"Language: {summary.language}",
    ]
    if summary.function_name:
        header_parts.append(f"Function: {summary.function_name}")
    if summary.parameters:
        header_parts.append("Parameters: " + ", ".join(summary.parameters))
    if summary.return_keywords:
        header_parts.append("Return clues: " + ", ".join(summary.return_keywords))
    if summary.raises:
        header_parts.append("Raises: " + ", ".join(summary.raises))
    if summary.called_functions:
        header_parts.append("Calls: " + ", ".join(summary.called_functions[:5]))
    header_parts.append(f"Complexity: {summary.complexity}")
    header_parts.append(
        "Write 1-2 sentences describing purpose, key parameters, return value, and any side effects or edge cases."
    )
    header_parts.append(
        "The comment must be actionable and non-trivial—avoid generic phrases like 'this function returns a value'."
    )

    header = "\n".join(header_parts)
    code_block = f"Code:\n```{summary.language}\n{code.strip()}\n```\n"
    return f"{header}\n{code_block}Comment:"


def score_comment(comment: str, summary: CodeSummary) -> Tuple[float, List[str]]:
    comment_lower = comment.lower()
    words = [w for w in re.split(r"[^a-zA-Z0-9_]+", comment_lower) if w]
    word_count = len(words)

    notes: List[str] = []

    # Coverage score
    anchor_tokens = summary.anchor_tokens
    coverage = (
        sum(1 for tok in anchor_tokens if tok in comment_lower) / max(len(anchor_tokens), 1)
        if anchor_tokens
        else 0.5
    )
    if anchor_tokens and coverage < 0.5:
        notes.append("Comment is missing key identifiers from the code.")

    # Length score: prefer 12-50 words
    if word_count == 0:
        length_score = 0.0
        notes.append("Comment is empty.")
    else:
        ideal_min, ideal_max = 12, 50
        if word_count < ideal_min:
            length_score = max(0.0, 1.0 - (ideal_min - word_count) / ideal_min)
            notes.append("Comment may be too short to be informative.")
        elif word_count > ideal_max:
            length_score = max(0.0, 1.0 - (word_count - ideal_max) / ideal_max)
            notes.append("Comment may be too long or verbose.")
        else:
            length_score = 1.0

    # Specificity score: look for verbs and mention of returns/side effects.
    specificity_terms = sum(
        1 for verb in KEYWORD_VERBS if verb in comment_lower
    )
    specificity_score = min(1.0, specificity_terms / 3.0)
    if specificity_score < 0.4:
        notes.append("Add more action verbs or describe behavior concretely.")

    quality = 0.5 * coverage + 0.3 * length_score + 0.2 * specificity_score
    return quality, notes

