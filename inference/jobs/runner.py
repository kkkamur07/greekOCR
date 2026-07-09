"""Inference runner shared by sync runs and queued inference jobs."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any

from inference.architectures.calamari import run_calamari_transcribe, run_calamari_transcribe_many
from inference.architectures.kraken import run_kraken_segment
from inference.contracts.common import InferenceTask, RegistryArchitecture
from inference.contracts.segment import SegmentRunResponse
from inference.contracts.transcribe import (
    TranscribeBatchLineResult,
    TranscribeBatchRunResponse,
    TranscribeLineRegion,
    TranscribeRunResponse,
)
from inference.infrastructure.settings import get_inference_settings
from PIL import Image

if TYPE_CHECKING:
    from inference.infrastructure.orm_models import InferenceJob
from inference.registry.resolve import resolve_registry_entry
from inference.weights import resolve_weights_source


def _crop_line_image(image_bytes: bytes, points: list[list[float]] | None) -> bytes:
    if not points:
        return image_bytes

    xs = [point[0] for point in points if len(point) == 2]
    ys = [point[1] for point in points if len(point) == 2]
    if not xs or not ys:
        return image_bytes

    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        left = max(0, int(min(xs)))
        top = max(0, int(min(ys)))
        right = min(width, int(max(xs)))
        bottom = min(height, int(max(ys)))
        if right <= left or bottom <= top:
            return image_bytes

        cropped = image.crop((left, top, right, bottom))
        output = BytesIO()
        cropped.save(output, format=image.format or "PNG")
        return output.getvalue()


def _line_regions_from_params(params: dict[str, Any] | None) -> list[TranscribeLineRegion]:
    raw_lines = (params or {}).get("lines")
    if raw_lines is None:
        return []
    if not isinstance(raw_lines, list):
        raise ValueError("transcribe params.lines must be a list")
    return [TranscribeLineRegion.model_validate(line) for line in raw_lines]


def run_model(
    *,
    task: InferenceTask,
    registry_model_id: str,
    registry_tag: str,
    image_bytes: bytes,
    params: dict[str, Any] | None = None,
) -> SegmentRunResponse | TranscribeRunResponse | TranscribeBatchRunResponse:
    settings = get_inference_settings()
    entry = resolve_registry_entry(
        registry_model_id=registry_model_id,
        registry_tag=registry_tag,
        task=task,
        registry_path=settings.inference_registry_path,
    )

    version = entry.versions[registry_tag]
    weights_path = resolve_weights_source(
        version.weights_source,
        registry_model_id=registry_model_id,
        registry_tag=registry_tag,
        architecture=entry.architecture.value,
    )

    if task == InferenceTask.segment:
        if entry.architecture == RegistryArchitecture.kraken_segment:
            return run_kraken_segment(
                image_bytes,
                model_path=weights_path,
                params=params,
            )
        raise ValueError(f"unsupported segment architecture: {entry.architecture.value}")

    if task == InferenceTask.transcribe:
        if entry.architecture == RegistryArchitecture.calamari:
            line_regions = _line_regions_from_params(params)
            if line_regions:
                outputs = run_calamari_transcribe_many(
                    [
                        _crop_line_image(image_bytes, region.points)
                        for region in line_regions
                    ],
                    checkpoint_path=weights_path,
                )
                return TranscribeBatchRunResponse(
                    lines=[
                        TranscribeBatchLineResult(
                            line_id=region.line_id,
                            line_index=region.line_index,
                            output=output,
                        )
                        for region, output in zip(line_regions, outputs, strict=True)
                    ]
                )
            return run_calamari_transcribe(
                image_bytes,
                checkpoint_path=weights_path,
            )
        raise ValueError(f"unsupported transcribe architecture: {entry.architecture.value}")

    raise ValueError(f"unsupported ML task for runner: {task.value}")


def run_job(
    job: InferenceJob,
) -> SegmentRunResponse | TranscribeRunResponse | TranscribeBatchRunResponse:
    return run_model(
        task=InferenceTask(job.task),
        registry_model_id=job.registry_model_id,
        registry_tag=job.registry_tag,
        image_bytes=job.image_bytes,
        params=job.params,
    )
