"""Application version helpers."""

from functools import lru_cache

from backend.core.settings._env import REPO_ROOT


@lru_cache
def get_version() -> str:
    return (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
