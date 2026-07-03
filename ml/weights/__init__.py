"""Weight source resolution and server-side cache layout."""

from __future__ import annotations

from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_ROOT = ML_ROOT / "weights"
DEFAULT_CACHE_ROOT = DEFAULT_WEIGHTS_ROOT / "cache"


def resolve_weights_source(uri: str, *, ml_root: Path = ML_ROOT) -> Path:
    if not uri.startswith("file://"):
        raise ValueError(f"unsupported weights source scheme: {uri}")
    relative = uri.removeprefix("file://")
    return (ml_root / relative).resolve()


def weight_cache_dir(
    registry_model_id: str,
    registry_tag: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    """Runtime cache path: weights/cache/<registry_model_id>/<tag>/."""
    return cache_root / registry_model_id / registry_tag


def bundled_weights_dir(
    registry_model_id: str,
    *,
    weights_root: Path = DEFAULT_WEIGHTS_ROOT,
) -> Path:
    """Bundled checkpoint layout under weights/<family>/<registry_model_id>/."""
    for family_dir in weights_root.iterdir():
        if not family_dir.is_dir() or family_dir.name == "cache":
            continue
        candidate = family_dir / registry_model_id
        if candidate.is_dir():
            return candidate
    return weights_root / registry_model_id
