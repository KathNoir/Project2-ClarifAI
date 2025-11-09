"""Utilities for validating Python code and extracting semantic tokens."""
import ast
import re
from typing import List


def extract_code_tokens(code: str) -> List[str]:
    """
    Extract semantic tokens from Python code using AST.
    
    Priority tokens:
    - Function names
    - Parameter names
    - Variable names
    - Class names
    - Called function names
    
    Args:
        code: Python code string
        
    Returns:
        List of important tokens (max 20)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _regex_extract(code)
    except Exception:
        return _regex_extract(code)

    tokens: list[str] = []
    visited: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name not in visited:
                tokens.append(node.name)
                visited.add(node.name)
            for arg in node.args.args:
                if arg.arg not in visited:
                    tokens.append(arg.arg)
                    visited.add(arg.arg)
        elif isinstance(node, ast.ClassDef) and node.name not in visited:
            tokens.append(node.name)
            visited.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id not in visited and len(node.id) > 1:
                tokens.append(node.id)
                visited.add(node.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in visited:
                tokens.append(node.func.id)
                visited.add(node.func.id)

    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen and len(token) > 1:
            seen.add(token)
            unique_tokens.append(token)

    return unique_tokens[:20]


def _regex_extract(code: str) -> List[str]:
    """
    Fallback token extraction using regex when AST parsing fails.
    
    Extracts function names, variable names, and parameter names.
    """
    identifiers = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
    python_keywords = {
        "def",
        "if",
        "for",
        "while",
        "return",
        "class",
        "import",
        "from",
        "as",
        "try",
        "except",
        "finally",
        "with",
        "pass",
        "continue",
        "break",
        "else",
        "elif",
        "lambda",
        "yield",
        "global",
        "nonlocal",
        "assert",
        "del",
        "and",
        "or",
        "not",
        "in",
        "is",
        "True",
        "False",
        "None",
        "self",
    }

    seen: set[str] = set()
    unique_tokens: list[str] = []
    for token in identifiers:
        if token in python_keywords or len(token) < 2:
            continue
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
        if len(unique_tokens) >= limit:
            break

    return unique_tokens[:20]


def validate_python_code(code: str) -> tuple[bool, str]:
    """
    Validate Python code syntax.
    
    Returns:
        (is_valid, error_message)
    """
    if not code or not code.strip():
        return False, "Code cannot be empty"

    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc.msg} (line {exc.lineno})"
    except Exception as exc:
        return False, f"Error parsing code: {exc}"

    return True, ""

