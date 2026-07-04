"""Weight source resolution and server-side cache layout."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_ROOT = ML_ROOT / "weights"
DEFAULT_CACHE_ROOT = Path(os.environ.get("ML_WEIGHTS_CACHE_DIR", DEFAULT_WEIGHTS_ROOT / "cache"))


def resolve_weights_source(uri: str, *, ml_root: Path = ML_ROOT) -> Path:
    if uri.startswith("package://"):
        package_resource = uri.removeprefix("package://")
        package_name, _, resource_name = package_resource.partition("/")
        if not package_name or not resource_name:
            raise ValueError("package weights source must be package://<package>/<resource>")
        resource = resources.files(package_name).joinpath(resource_name)
        if not resource.is_file():
            raise FileNotFoundError(f"package weights source not found: {uri}")
        return Path(str(resource))

    if not uri.startswith("file://"):
        raise ValueError(f"unsupported weights source scheme: {uri}")

    relative = uri.removeprefix("file://")
    source_path = Path(relative)
    if source_path.is_absolute():
        raise ValueError("file weights source must be relative to ML_ROOT")

    resolved_root = ml_root.resolve()
    resolved_path = (resolved_root / source_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("file weights source must stay within ML_ROOT") from exc

    return resolved_path


def weight_cache_dir(
    registry_model_id: str,
    registry_tag: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    """Runtime cache path: weights/cache/<registry_model_id>/<tag>/."""
    return cache_root / registry_model_id / registry_tag


def bundled_weights_dir(
    weights_source: str,
    *,
    ml_root: Path = ML_ROOT,
) -> Path:
    """Bundled checkpoint directory resolved directly from registry weights_source."""
    return resolve_weights_source(weights_source, ml_root=ml_root).parent
