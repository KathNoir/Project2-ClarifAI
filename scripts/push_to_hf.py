from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import HfApi, create_repo

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUDES = {
    ".git",
    ".gitignore",
    ".gitattributes",
    ".venv",
    "__pycache__",
    "reports",
}


def _should_skip(path: Path, excludes: Iterable[str]) -> bool:
    relative = path.relative_to(REPO_ROOT)
    for part in relative.parts:
        if part in excludes:
            return True
    return False


def _gather_files(excludes: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if path.is_file() and not _should_skip(path, excludes):
            files.append(path)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the Clarify.dev workspace to a Hugging Face Space.",
    )
    parser.add_argument(
        "--space-id",
        required=True,
        help="Target Space in the form <username>/<space-name>.",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token. Defaults to HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--commit-message",
        default="Automated push from run_regression pipeline",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=(),
        help="Additional paths to exclude from the upload.",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Target branch on the Space (default: main).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Provide --token or set HF_TOKEN.")

    excludes = set(DEFAULT_EXCLUDES) | set(args.exclude or ())
    files = _gather_files(excludes)

    api = HfApi(token=token)
    create_repo(
        repo_id=args.space_id,
        token=token,
        repo_type="space",
        private=False,
        exist_ok=True,
        space_sdk="gradio",
    )

    operations = []
    for file_path in files:
        rel_path = file_path.relative_to(REPO_ROOT)
        with file_path.open("rb") as handle:
            operations.append(
                {
                    "path_in_repo": str(rel_path).replace("\\", "/"),
                    "path_or_fileobj": handle.read(),
                }
            )

    api.upload_files(
        repo_id=args.space_id,
        repo_type="space",
        commit_message=args.commit_message,
        files=[(op["path_in_repo"], op["path_or_fileobj"]) for op in operations],
        revision=args.branch,
    )
    print(f"Pushed {len(files)} files to {args.space_id}@{args.branch}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

