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


def _is_weights_cached(
    weights_source: str,
    *,
    registry_model_id: str,
    registry_tag: str,
    hub_revision: str | None,
    artifact_sha256: str | None,
    architecture: str,
) -> bool:
    """Return True when the model weights are already available locally.

    Only inspects local disk; never contacts the Hub. Non-hf sources (bundled,
    package, file) ship with the helper and are always considered cached.
    """
    if not weights_source.startswith("hf://"):
        return True

    from src.hf.resolve.artifacts import find_hub_artifact, verify_artifact_sha256
    from src.hf.resolve.cache import cache_dir_for
    from src.hf.resolve.manifest import load_manifest, manifest_matches_expected

    cache_dir = cache_dir_for(registry_model_id, registry_tag)
    manifest = load_manifest(cache_dir)
    if (
        manifest is None
        or hub_revision is None
        or artifact_sha256 is None
        or not manifest_matches_expected(
            manifest,
            repo_id=weights_source.removeprefix("hf://").rsplit("@", 1)[0],
            hub_revision=hub_revision,
            artifact_sha256=artifact_sha256,
        )
    ):
        return False
    try:
        artifact = find_hub_artifact(cache_dir, architecture=architecture)
        if str(artifact.relative_to(cache_dir)) != manifest.artifact_path:
            return False
        verify_artifact_sha256(artifact, artifact_sha256)
    except (FileNotFoundError, ValueError):
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
        hub_revision=version.hub_revision,
        artifact_sha256=version.artifact_sha256,
        architecture=entry.architecture.value,
    )
    return CacheStatusResponse(
        registry_model_id=registry_model_id,
        registry_tag=registry_tag,
        cached=cached,
    )
