"""Local weights cache status for the Inference helper (no network calls)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from inference.helper.settings import get_helper_settings
from inference.registry import get_model_entry, load_registry

router = APIRouter(prefix="/inference/v1", tags=["cache"])


class CacheStatusResponse(BaseModel):
    registry_model_id: str
    registry_tag: str
    cached: bool


def _is_weights_cached(weights_source: str, *, registry_model_id: str, registry_tag: str, architecture: str) -> bool:
    """Return True when the model weights are already available locally.

    Only inspects local disk; never contacts the Hub. Non-hf sources (bundled,
    package, file) ship with the helper and are always considered cached.
    """
    if not weights_source.startswith("hf://"):
        return True

    from src.hf.resolve.artifacts import find_hub_artifact
    from src.hf.resolve.cache import cache_dir_for
    from src.hf.resolve.manifest import load_manifest

    cache_dir = cache_dir_for(registry_model_id, registry_tag)
    if load_manifest(cache_dir) is None:
        return False
    try:
        find_hub_artifact(cache_dir, architecture=architecture)
    except FileNotFoundError:
        return False
    return True


@router.get(
    "/cache-status",
    response_model=CacheStatusResponse,
    status_code=status.HTTP_200_OK,
)
def cache_status(registry_model_id: str, registry_tag: str = "stable") -> CacheStatusResponse:
    registry = load_registry(get_helper_settings().inference_registry_path)
    try:
        entry = get_model_entry(registry, registry_model_id, registry_tag)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    version = entry.versions[registry_tag]
    cached = _is_weights_cached(
        version.weights_source,
        registry_model_id=registry_model_id,
        registry_tag=registry_tag,
        architecture=entry.architecture.value,
    )
    return CacheStatusResponse(
        registry_model_id=registry_model_id,
        registry_tag=registry_tag,
        cached=cached,
    )
