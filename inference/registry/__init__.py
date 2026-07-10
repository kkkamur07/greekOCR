"""Registry model catalog loaded from registry.yaml."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from inference.contracts.common import (
    ComputeDevice,
    HostEligibility,
    InferenceTask,
    RegistryArchitecture,
)

INFERENCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = INFERENCE_ROOT / "registry.yaml"
_COMMIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RegistryVersionEntry(BaseModel):
    weights_source: str = Field(min_length=1)
    hub_revision: str | None = None
    artifact_sha256: str | None = None

    @field_validator("hub_revision")
    @classmethod
    def validate_hub_revision(cls, value: str | None) -> str | None:
        if value is not None and not _COMMIT_SHA_PATTERN.fullmatch(value):
            raise ValueError("hub_revision must be a 40-character lowercase commit SHA")
        return value

    @field_validator("artifact_sha256")
    @classmethod
    def validate_artifact_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("artifact_sha256 must be a 64-character lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_hf_provenance_pair(self) -> RegistryVersionEntry:
        if self.weights_source.startswith("hf://") and (
            bool(self.hub_revision) != bool(self.artifact_sha256)
        ):
            raise ValueError("hf weights_source must provide both hub_revision and artifact_sha256")
        if not self.weights_source.startswith("hf://") and self.hub_revision is not None:
            raise ValueError("hub_revision is only valid for hf weights_source")
        return self


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
