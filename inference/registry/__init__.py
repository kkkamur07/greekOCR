"""Registry model catalog loaded from registry.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from inference.contracts.common import ComputeDevice, HostEligibility, InferenceTask, RegistryArchitecture

INFERENCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = INFERENCE_ROOT / "registry.yaml"


class RegistryVersionEntry(BaseModel):
    weights_source: str = Field(min_length=1)


class RegistryModelEntry(BaseModel):
    task: InferenceTask
    architecture: RegistryArchitecture
    device: ComputeDevice
    host_eligibility: HostEligibility = HostEligibility.local
    versions: dict[str, RegistryVersionEntry] = Field(min_length=1)


class RegistryDocument(BaseModel):
    models: dict[str, RegistryModelEntry] = Field(min_length=1)


def load_registry(path: Path | None = None) -> RegistryDocument:
    registry_path = path or DEFAULT_REGISTRY_PATH
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

    return RegistryDocument.model_validate(raw)


def get_model_entry(
    registry: RegistryDocument,
    registry_model_id: str,
    registry_tag: str = "stable",
) -> RegistryModelEntry:
    try:
        model = registry.models[registry_model_id]
    except KeyError as exc:
        raise KeyError(f"unknown registry model id: {registry_model_id}") from exc

    try:
        model.versions[registry_tag]
    except KeyError as exc:
        raise KeyError(
            f"unknown registry tag {registry_tag!r} for model {registry_model_id!r}"
        ) from exc
    return model
