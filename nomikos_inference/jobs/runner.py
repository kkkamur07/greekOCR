"""Model execution for the synchronous run path."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from nomikos_inference.admission import validate_image_bytes, validate_request_params
from nomikos_inference.architectures.blla import run_blla_segment
from nomikos_inference.architectures.calamari import (
    TranscribeLineFailure,
    run_calamari_transcribe,
    run_calamari_transcribe_many,
)
from nomikos_inference.architectures.calamari.preprocessing import (
    TRAINING_CROP_PADDING,
    crop_line,
)
from nomikos_inference.contracts.common import InferenceTask, LineCrop, RegistryArchitecture
from nomikos_inference.contracts.segment import SegmentRunResponse
from nomikos_inference.contracts.transcribe import (
    TRANSCRIBE_LINE_ERROR,
    TranscribeBatchLineResult,
    TranscribeBatchRunResponse,
    TranscribeLineRegion,
    TranscribeRunResponse,
)
from nomikos_inference.registry.resolve import resolve_registry_entry
from nomikos_inference.settings import get_inference_settings
from nomikos_inference.weights import resolve_weights_source

logger = logging.getLogger(__name__)


#: ``save_crop`` in ``src/preprocessing_data/syriac/xml_to_data.py`` wrote every
#: training crop as a grayscale JPEG with exactly these options, and the
#: trainer's loader read that file back. The lossy round trip is therefore part
#: of the picture the model was fitted on, not an artifact of storage, so serving
#: reproduces it rather than handing over a cleaner PNG.
TRAINING_CROP_JPEG_OPTIONS = {"format": "JPEG", "quality": 82, "optimize": True}


def _crop_line_image(
    image: Image.Image,
    image_bytes: bytes,
    points: list[list[float]] | None,
    line_crop: LineCrop = LineCrop.polygon_white,
    line_crop_padding: int = TRAINING_CROP_PADDING,
) -> bytes:
    """Cut one line out of a page the way *this model's* training crops were cut.

    One crop function serves every model: the polygon's box widened by
    ``line_crop_padding``, clamped to the page, with everything outside the
    polygon painted white, which is what
    ``src/preprocessing_data/syriac/xml_to_data.py::crop_polygon`` wrote. Only
    the padding differs, and it differs enough to matter: Greek was exported at 0
    and reads 125 of its 204 corpus lines exactly there against 0 of 204 at 12.
    So the registry states both per model and they are threaded down to here.

    Takes the open ``image`` so a page with N lines is decoded once, not N times:
    ``Image.crop`` forces a full decode of the source on every call, so re-opening
    the multi-megapixel scan per line was O(N) full decodes of the same bytes.
    ``image_bytes`` is still returned verbatim for the whole-page fallback so the
    downstream model sees the original encoding, not a re-encode.

    The crop goes on as a mode ``L`` JPEG at the exporter's own quality, not as a
    lossless PNG: the model was trained on crops that had been through that
    encoder, so a clean PNG would hand it pixels slightly unlike any it ever saw.
    A ``ValueError`` from the crop (degenerate geometry off the page) propagates:
    ``_transcribe_batch`` isolates it to its own line.
    """
    if not points:
        return image_bytes

    polygon = [point for point in points if len(point) == 2]
    if not polygon:
        return image_bytes

    cropped = crop_line(image, polygon, line_crop, padding=line_crop_padding)
    output = BytesIO()
    Image.fromarray(cropped, mode="L").save(output, **TRAINING_CROP_JPEG_OPTIONS)
    return output.getvalue()


def _line_regions_from_params(params: dict[str, Any] | None) -> list[TranscribeLineRegion]:
    raw_lines = (params or {}).get("lines")
    if raw_lines is None:
        return []
    if not isinstance(raw_lines, list):
        raise ValueError("transcribe params.lines must be a list")
    return [TranscribeLineRegion.model_validate(line) for line in raw_lines]


def _transcribe_batch(
    image_bytes: bytes,
    line_regions: list[TranscribeLineRegion],
    *,
    checkpoint_path: Path,
    artifact_sha256: str | None,
    line_crop: LineCrop = LineCrop.polygon_white,
    line_crop_padding: int = TRAINING_CROP_PADDING,
) -> TranscribeBatchRunResponse:
    """Transcribe every requested line, keeping per-line failures per-line.

    A malformed polygon or an undecodable crop costs its own line only: the
    other lines of the page still come back with their text, and the bad one
    carries ``error`` instead of ``output``. The failure is logged here rather
    than deeper down because this is the only layer that knows which document
    line the caller meant.
    """
    crops: list[bytes] = []
    cropped_positions: list[int] = []
    errors: dict[int, str] = {}

    # Decode the page once and crop every line from it. Re-opening the scan per
    # line re-decoded the whole multi-megapixel image N times for an N-line page.
    with Image.open(BytesIO(image_bytes)) as page_image:
        page_image.load()
        for position, region in enumerate(line_regions):
            try:
                crop = _crop_line_image(
                    page_image,
                    image_bytes,
                    region.points,
                    line_crop=line_crop,
                    line_crop_padding=line_crop_padding,
                )
            except Exception as error:  # noqa: BLE001 - one bad region is not a bad page
                logger.warning(
                    "transcribe line crop failed (line_index=%s, line_id=%s)",
                    region.line_index,
                    region.line_id,
                    exc_info=error,
                )
                errors[position] = TRANSCRIBE_LINE_ERROR
                continue
            crops.append(crop)
            cropped_positions.append(position)

    if not crops:
        # No line even reached the model. Nothing here is worth returning as a
        # partial success, and the cause is the caller's geometry.
        raise ValueError("no transcribable line regions in request")

    outcomes = run_calamari_transcribe_many(
        crops,
        checkpoint_path=checkpoint_path,
        artifact_sha256=artifact_sha256,
    )

    outputs: dict[int, TranscribeRunResponse] = {}
    for position, outcome in zip(cropped_positions, outcomes, strict=True):
        region = line_regions[position]
        if isinstance(outcome, TranscribeLineFailure):
            logger.warning(
                "transcribe line failed (line_index=%s, line_id=%s)",
                region.line_index,
                region.line_id,
                exc_info=outcome.error,
            )
            errors[position] = TRANSCRIBE_LINE_ERROR
            continue
        outputs[position] = outcome

    # ``run_calamari_transcribe_many`` re-raises when every line it was given
    # failed, so at least one output survives here and the response can never be
    # an all-error batch dressed up as a success.
    return TranscribeBatchRunResponse(
        lines=[
            TranscribeBatchLineResult(
                line_id=region.line_id,
                line_index=region.line_index,
                output=outputs.get(position),
                error=errors.get(position),
            )
            for position, region in enumerate(line_regions)
        ]
    )


def run_model(
    *,
    task: InferenceTask,
    registry_model_id: str,
    registry_tag: str,
    image_bytes: bytes,
    params: dict[str, Any] | None = None,
) -> SegmentRunResponse | TranscribeRunResponse | TranscribeBatchRunResponse:
    settings = get_inference_settings()
    validate_image_bytes(image_bytes, settings)
    validate_request_params(params or {}, settings)
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
        hub_revision=version.hub_revision,
        artifact_sha256=version.artifact_sha256,
        architecture=entry.architecture.value,
    )

    if task == InferenceTask.segment:
        if entry.architecture in {
            RegistryArchitecture.blla,
            RegistryArchitecture.blla_segment,
        }:
            return run_blla_segment(
                image_bytes,
                model_path=weights_path,
                artifact_sha256=version.artifact_sha256,
                params=params,
            )
        raise ValueError(f"unsupported segment architecture: {entry.architecture.value}")

    if task == InferenceTask.transcribe:
        if entry.architecture == RegistryArchitecture.calamari:
            line_regions = _line_regions_from_params(params)
            if line_regions:
                return _transcribe_batch(
                    image_bytes,
                    line_regions,
                    checkpoint_path=weights_path,
                    artifact_sha256=version.artifact_sha256,
                    # The registry requires both on every transcribe entry; the
                    # fallbacks only cover an entry built outside the loader.
                    line_crop=entry.line_crop or LineCrop.polygon_white,
                    line_crop_padding=(
                        TRAINING_CROP_PADDING
                        if entry.line_crop_padding is None
                        else entry.line_crop_padding
                    ),
                )
            return run_calamari_transcribe(
                image_bytes,
                checkpoint_path=weights_path,
                artifact_sha256=version.artifact_sha256,
            )
        raise ValueError(f"unsupported transcribe architecture: {entry.architecture.value}")

    raise ValueError(f"unsupported ML task for runner: {task.value}")
