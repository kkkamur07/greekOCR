"""ML job runners backed by registry model entries."""

from __future__ import annotations

from ml.architectures.kraken import run_kraken_segment
from ml.contracts.common import MLTask
from ml.contracts.segment import SegmentRunResponse
from ml.contracts.transcribe import CharacterConfidence, TranscribeRunResponse
from ml.infrastructure.orm_models import MLJob
from ml.infrastructure.settings import get_ml_settings
from ml.registry import get_model_entry, load_registry
from ml.weights import resolve_weights_source


def run_segment(
    *,
    registry_model_id: str,
    registry_tag: str,
    image_bytes: bytes,
) -> SegmentRunResponse:
    settings = get_ml_settings()
    registry = load_registry(settings.ml_registry_path)
    entry = get_model_entry(registry, registry_model_id, registry_tag)
    if entry.task != MLTask.segment:
        raise ValueError(
            f"registry model {registry_model_id!r} is {entry.task.value!r}, not 'segment'"
        )

    weights_source = entry.versions[registry_tag].weights_source
    weights_path = resolve_weights_source(weights_source)
    return run_kraken_segment(image_bytes, weights_path=weights_path)


def run_job(job: MLJob) -> SegmentRunResponse | TranscribeRunResponse:
    if job.task == MLTask.segment:
        return run_segment(
            registry_model_id=job.registry_model_id,
            registry_tag=job.registry_tag,
            image_bytes=job.image_bytes,
        )
    if job.task == MLTask.transcribe:
        text = f"mock:{len(job.image_bytes)}"
        return TranscribeRunResponse(
            text=text,
            confidence=1.0,
            character_confidences=[
                CharacterConfidence(char=char, confidence=1.0) for char in text
            ],
        )
    raise ValueError(f"unsupported ML task for runner: {job.task}")
