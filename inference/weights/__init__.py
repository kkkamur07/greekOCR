"""Weight source resolution and server-side cache layout."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[1]
from src.hf.paths import HF_ROOT
DEFAULT_WEIGHTS_ROOT = INFERENCE_ROOT / "weights"
DEFAULT_CACHE_ROOT = Path(
    os.environ.get("INFERENCE_WEIGHTS_CACHE_DIR", DEFAULT_WEIGHTS_ROOT / "cache")
)
LOCAL_BUNDLED_PREFIX = "local/"


def resolve_weights_source(
    uri: str,
    *,
    inference_root: Path = INFERENCE_ROOT,
    registry_model_id: str | None = None,
    registry_tag: str | None = None,
    architecture: str | None = None,
) -> Path:
    if uri.startswith("hf://"):
        if not registry_model_id or not registry_tag:
            raise ValueError("hf weights source requires registry_model_id and registry_tag")
        from src.hf.resolve import resolve_hf_weights_source

        return resolve_hf_weights_source(
            uri,
            registry_model_id=registry_model_id,
            registry_tag=registry_tag,
            architecture=architecture,
        )

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
        raise ValueError("file weights source must be relative to INFERENCE_ROOT or src/hf/")

    if relative.startswith(LOCAL_BUNDLED_PREFIX):
        resolved_root = HF_ROOT.resolve()
        resolved_path = (resolved_root / source_path).resolve()
        root_label = "src/hf/"
    else:
        resolved_root = inference_root.resolve()
        resolved_path = (resolved_root / source_path).resolve()
        root_label = "INFERENCE_ROOT"

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"file weights source must stay within {root_label}") from exc

    return resolved_path
