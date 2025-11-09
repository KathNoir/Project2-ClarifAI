"""
Minimal copy of the CodeSearchNet preprocessing utilities used in the Clarify.dev project.
The implementation is trimmed slightly to stay lightweight for Spaces demos.
"""

from __future__ import annotations

import ast
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


class CodeSearchNetPreprocessor:
    def __init__(self, data_dir: str = "./codesearchnet_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def download_dataset(self, language: str = "python", max_samples: int | None = None):
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled by HTTP layer
            raise RuntimeError(
                "Hugging Face 'datasets' package is required. Add it to requirements.txt."
            ) from exc

        dataset = load_dataset("code_search_net", language)

        collected: List[Dict[str, str]] = []
        for split in ["train", "validation", "test"]:
            for item in dataset[split]:
                if max_samples is not None and len(collected) >= max_samples:
                    break
                doc = item.get("func_documentation_string") or ""
                code = item.get("func_code_string") or ""
                if not doc or not code:
                    continue
                collected.append(
                    {
                        "code": code,
                        "comment": doc,
                        "func_name": item.get("func_name"),
                        "split": split,
                    }
                )
        return collected

    def extract_code_tokens(self, code: str, limit: int = 20) -> List[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._regex_extract(code, limit=limit)

        tokens: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                tokens.append(node.name)
                tokens.extend(arg.arg for arg in node.args.args)
            elif isinstance(node, ast.ClassDef):
                tokens.append(node.name)
            elif isinstance(node, ast.Name):
                tokens.append(node.id)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                tokens.append(node.func.id)

        unique: List[str] = []
        seen = set()
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique.append(token)
                if len(unique) >= limit:
                    break
        return unique

    def _regex_extract(self, code: str, limit: int = 20) -> List[str]:
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

        unique: List[str] = []
        seen = set()
        for token in identifiers:
            if token in python_keywords or len(token) < 2:
                continue
            if token not in seen:
                seen.add(token)
                unique.append(token)
                if len(unique) >= limit:
                    break
        return unique

    def create_train_dev_test_split(
        self, data: List[Dict], train_ratio: float = 0.8, dev_ratio: float = 0.1, test_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6

        shuffled = data.copy()
        random.seed(42)
        random.shuffle(shuffled)

        total = len(shuffled)
        train_end = int(total * train_ratio)
        dev_end = train_end + int(total * dev_ratio)

        return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]



