"""Resolve architecture-specific runners from registry entries."""

from __future__ import annotations

import os
from typing import Protocol

from ml.architectures.mock import MockTranscribeRunner
from ml.contracts.common import MLTask, RegistryArchitecture
from ml.contracts.transcribe import TranscribeRunResponse
from ml.registry import RegistryModelEntry, get_model_entry, load_registry


class TranscribeRunner(Protocol):
    def transcribe(self, image_bytes: bytes, *, params: dict) -> TranscribeRunResponse: ...


def resolve_transcribe_runner(
    entry: RegistryModelEntry,
    *,
    registry_tag: str = "stable",
) -> TranscribeRunner:
    if os.environ.get("ML_FORCE_MOCK_RUNNER", "").lower() in {"1", "true", "yes"}:
        return MockTranscribeRunner()

    if entry.architecture == RegistryArchitecture.calamari:
        from ml.architectures.calamari import CalamariTranscribeRunner

        version = entry.versions[registry_tag]
        return CalamariTranscribeRunner(entry, version=version)

    return MockTranscribeRunner()


def transcribe_from_registry(
    *,
    registry_model_id: str,
    registry_tag: str,
    image_bytes: bytes,
    params: dict,
) -> TranscribeRunResponse:
    registry = load_registry()
    entry = get_model_entry(registry, registry_model_id, registry_tag)
    if entry.task != MLTask.transcribe:
        raise ValueError(f"registry model {registry_model_id!r} is not a transcribe task")
    runner = resolve_transcribe_runner(entry, registry_tag=registry_tag)
    return runner.transcribe(image_bytes, params=params)
