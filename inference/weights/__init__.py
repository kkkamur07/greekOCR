"""Weight source resolution and server-side cache layout."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_ROOT = INFERENCE_ROOT / "weights"
DEFAULT_CACHE_ROOT = Path(os.environ.get("INFERENCE_WEIGHTS_CACHE_DIR", DEFAULT_WEIGHTS_ROOT / "cache"))


def resolve_weights_source(uri: str, *, inference_root: Path = INFERENCE_ROOT) -> Path:
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
        raise ValueError("file weights source must be relative to INFERENCE_ROOT")

    resolved_root = inference_root.resolve()
    resolved_path = (resolved_root / source_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("file weights source must stay within INFERENCE_ROOT") from exc

    return resolved_path
