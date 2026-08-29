"""Resolve nomikos app root for platform scripts (repo layout or API container)."""

from __future__ import annotations

import sys
from pathlib import Path

_PLATFORM_DIR = Path(__file__).resolve().parent


def nomikos_root() -> Path:
    repo_or_app = _PLATFORM_DIR.parents[1]
    nested = repo_or_app / "nomikos"
    if (nested / "backend").is_dir():
        return nested
    if (repo_or_app / "backend").is_dir():
        return repo_or_app
    raise RuntimeError(
        "Could not locate nomikos app root (expected nomikos/backend or /app/backend)"
    )


def ensure_nomikos_on_path() -> Path:
    root = nomikos_root()
    paths_to_add: list[Path] = [root]
    if not (root / "nomikos_inference").is_dir() and (root.parent / "nomikos_inference").is_dir():
        paths_to_add.append(root.parent)
    for path in paths_to_add:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root
