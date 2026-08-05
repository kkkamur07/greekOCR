"""The helper's one capability document: who it is and what it can run."""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel

from inference.contracts.common import HostEligibility, InferenceTask
from inference.helper.settings import HELPER_VERSION, get_helper_settings
from inference.registry import load_registry

router = APIRouter(prefix="/inference/v1", tags=["info"])

# Identifies this process as the Nomicous helper. A browser that finds *some*
# server on 127.0.0.1:8001 must be able to tell it apart from an unrelated one
# before POSTing a manuscript image to it.
HELPER_SERVICE_NAME = "nomicous-inference-helper"


class InfoModel(BaseModel):
    registry_model_id: str
    task: InferenceTask
    host_eligibility: HostEligibility
    tags: list[str]
    cached: bool


class InfoResponse(BaseModel):
    service: str
    version: str
    models: list[InfoModel]


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
    package, file) ship with the helper, but their presence is still verified.
    Digest verification is memoized in ``src.hf.resolve.artifacts``, so this
    reuses the hash a run of the same artifact already computed.
    """
    if not weights_source.startswith("hf://"):
        from inference.weights import resolve_weights_source

        try:
            resolve_weights_source(
                weights_source,
                registry_model_id=registry_model_id,
                registry_tag=registry_tag,
                hub_revision=hub_revision,
                artifact_sha256=artifact_sha256,
                architecture=architecture,
            )
        except (FileNotFoundError, ValueError):
            return False
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


@router.get("/info", response_model=InfoResponse, status_code=status.HTTP_200_OK)
def info() -> InfoResponse:
    """Everything a client needs to decide whether to send work here."""
    registry = load_registry(get_helper_settings().inference_registry_path)
    models = [
        InfoModel(
            registry_model_id=model_id,
            task=entry.task,
            host_eligibility=entry.host_eligibility,
            tags=sorted(entry.versions),
            cached=all(
                _is_weights_cached(
                    version.weights_source,
                    registry_model_id=model_id,
                    registry_tag=tag,
                    hub_revision=version.hub_revision,
                    artifact_sha256=version.artifact_sha256,
                    architecture=entry.architecture.value,
                )
                for tag, version in entry.versions.items()
            ),
        )
        for model_id, entry in sorted(registry.models.items())
    ]
    return InfoResponse(
        service=HELPER_SERVICE_NAME,
        version=HELPER_VERSION,
        models=models,
    )
