"""Inference runner shared by sync runs and queued ML jobs."""

from __future__ import annotations

from typing import Any

from ml_service.architectures.calamari import run_calamari_transcribe
from ml_service.architectures.kraken import run_kraken_segment
from ml_service.contracts.common import MLTask, RegistryArchitecture
from ml_service.contracts.segment import SegmentRunResponse
from ml_service.contracts.transcribe import TranscribeRunResponse
from ml_service.infrastructure.orm_models import MLJob
from ml_service.infrastructure.settings import get_ml_settings
from ml_service.registry import get_model_entry, load_registry
from ml_service.weights import resolve_weights_source


def run_model(
    *,
    task: MLTask,
    registry_model_id: str,
    registry_tag: str,
    image_bytes: bytes,
    params: dict[str, Any] | None = None,
) -> SegmentRunResponse | TranscribeRunResponse:
    settings = get_ml_settings()
    registry = load_registry(settings.ml_registry_path)
    entry = get_model_entry(registry, registry_model_id, registry_tag)
    if entry.task != task:
        raise ValueError(f"registry model task does not match {task.value} request")

    version = entry.versions[registry_tag]
    weights_path = resolve_weights_source(version.weights_source)

    if task == MLTask.segment:
        if entry.architecture == RegistryArchitecture.kraken_segment:
            return run_kraken_segment(
                image_bytes,
                model_path=weights_path,
            )
        raise ValueError(f"unsupported segment architecture: {entry.architecture.value}")

    if task == MLTask.transcribe:
        if entry.architecture == RegistryArchitecture.calamari:
            return run_calamari_transcribe(
                image_bytes,
                checkpoint_path=weights_path,
            )
        raise ValueError(f"unsupported transcribe architecture: {entry.architecture.value}")

    raise ValueError(f"unsupported ML task for runner: {task.value}")


def run_job(job: MLJob) -> SegmentRunResponse | TranscribeRunResponse:
    return run_model(
        task=MLTask(job.task),
        registry_model_id=job.registry_model_id,
        registry_tag=job.registry_tag,
        image_bytes=job.image_bytes,
        params=job.params,
    )
