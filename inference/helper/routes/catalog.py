"""Registry catalog for the Inference helper."""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel

from inference.contracts.common import (
    ComputeDevice,
    HostEligibility,
    InferenceTask,
    RegistryArchitecture,
)
from inference.helper.settings import get_helper_settings
from inference.registry import load_registry

router = APIRouter(prefix="/inference/v1", tags=["catalog"])


class CatalogModelResponse(BaseModel):
    registry_model_id: str
    task: InferenceTask
    architecture: RegistryArchitecture
    device: ComputeDevice
    host_eligibility: HostEligibility
    tags: list[str]


class CatalogResponse(BaseModel):
    models: list[CatalogModelResponse]


@router.get("/catalog", response_model=CatalogResponse, status_code=status.HTTP_200_OK)
def catalog() -> CatalogResponse:
    registry = load_registry(get_helper_settings().inference_registry_path)
    models = [
        CatalogModelResponse(
            registry_model_id=model_id,
            task=entry.task,
            architecture=entry.architecture,
            device=entry.device,
            host_eligibility=entry.host_eligibility,
            tags=sorted(entry.versions.keys()),
        )
        for model_id, entry in sorted(registry.models.items())
    ]
    return CatalogResponse(models=models)
