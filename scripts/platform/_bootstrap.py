"""Resolve nomicous app root for platform scripts (repo layout or API container)."""

from __future__ import annotations

import sys
from pathlib import Path

_PLATFORM_DIR = Path(__file__).resolve().parent


def nomicous_root() -> Path:
    repo_or_app = _PLATFORM_DIR.parents[1]
    nested = repo_or_app / "nomicous"
    if (nested / "backend").is_dir():
        return nested
    if (repo_or_app / "backend").is_dir():
        return repo_or_app
    raise RuntimeError(
        "Could not locate nomicous app root (expected nomicous/backend or /app/backend)"
    )


def ensure_nomicous_on_path() -> Path:
    root = nomicous_root()
    paths_to_add: list[Path] = [root]
    if not (root / "inference").is_dir() and (root.parent / "inference").is_dir():
        paths_to_add.append(root.parent)
    for path in paths_to_add:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root
