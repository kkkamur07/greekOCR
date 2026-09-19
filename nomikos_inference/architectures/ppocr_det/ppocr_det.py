"""PP-OCRv6 detection segmentation on the ONNX Runtime CPU runtime (ADR 0006).

Runs a PP-OCRv6 text detection graph (a DB detector: normalised image in,
probability map out) and returns the same segment contract the kraken blla
segmenter returns. ``resolve_artifact`` verifies the **artifact SHA-256**
before the file is opened.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from nomikos_inference.admission import open_image_bytes
from nomikos_inference.architectures.artifact import ArtifactHandle, resolve_artifact
from nomikos_inference.architectures.ppocr_det.postprocessing import detect_lines
from nomikos_inference.architectures.ppocr_det.preprocessing import preprocess_ppocr_det_image
from nomikos_inference.architectures.ppocr_det.response import (
    DEFAULT_BASELINE_FRACTION,
    build_ppocr_det_response,
)
from nomikos_inference.contracts.segment import SegmentRunResponse

PPOCR_DET_ARTIFACT_SUFFIXES = frozenset({".onnx"})

DEFAULT_LIMIT_SIDE_LEN = 1920
MIN_LIMIT_SIDE_LEN = 320
MAX_LIMIT_SIDE_LEN = 4000
DEFAULT_THRESH = 0.2
DEFAULT_BOX_THRESH = 0.45
DEFAULT_UNCLIP_RATIO = 1.4
DEFAULT_MAX_CANDIDATES = 3000


class PPOCRDetUnavailableError(RuntimeError):
    """Raised when a PP-OCRv6 detection runtime artifact cannot be used."""


def _resolve_ppocr_det_artifact(
    model_path: Path,
    artifact_sha256: str | None = None,
) -> ArtifactHandle:
    return resolve_artifact(
        model_path,
        label="PP-OCRv6 det model",
        allowed_suffixes=PPOCR_DET_ARTIFACT_SUFFIXES,
        unusable_error=PPOCRDetUnavailableError,
        unusable_message=f"PP-OCRv6 det runtime requires an .onnx model: {model_path}",
        artifact_sha256=artifact_sha256,
    )


@lru_cache(maxsize=4)
def _load_ppocr_det_session(
    model_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[Any, str, str]:
    """Open a session and read the tensor names this decoder needs.

    ``fingerprint`` is part of the cache key rather than an argument the loader
    reads: it is what makes a *replaced* artifact file miss the cache instead of
    serving the previous model for the life of the process.
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise PPOCRDetUnavailableError("PP-OCRv6 det ONNX graph must have one input and output")
        if len(inputs[0].shape) != 4 or len(outputs[0].shape) != 4:
            raise PPOCRDetUnavailableError("PP-OCRv6 det ONNX graph must use 4D tensors")
        return session, inputs[0].name, outputs[0].name
    except PPOCRDetUnavailableError:
        raise
    except ImportError as error:
        raise PPOCRDetUnavailableError(
            "onnxruntime is required for the PP-OCRv6 det runtime"
        ) from error
    except Exception as error:
        raise PPOCRDetUnavailableError("unable to load PP-OCRv6 det ONNX model") from error


def _positive_float_param(params: Mapping[str, Any], key: str, default: float) -> float:
    """Parse a caller-supplied positive number, falling back to the default.

    Like the blla helper of the same shape: upper bounds belong to admission
    where they exist, and an unparseable or non-positive value means "use the
    default" rather than failing the page.
    """
    value = params.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _limit_side_len(params: Mapping[str, Any]) -> int:
    value = params.get("limit_side_len", DEFAULT_LIMIT_SIDE_LEN)
    if isinstance(value, bool):
        raise ValueError("limit_side_len must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("limit_side_len must be an integer") from None
    if isinstance(value, float) and float(parsed) != value:
        raise ValueError("limit_side_len must be an integer")
    if parsed < MIN_LIMIT_SIDE_LEN or parsed > MAX_LIMIT_SIDE_LEN:
        raise ValueError(
            f"limit_side_len must be between {MIN_LIMIT_SIDE_LEN} and {MAX_LIMIT_SIDE_LEN}"
        )
    return parsed


def _max_candidates(params: Mapping[str, Any]) -> int:
    value = params.get("max_candidates", DEFAULT_MAX_CANDIDATES)
    if isinstance(value, bool):
        raise ValueError("max_candidates must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("max_candidates must be an integer") from None
    if isinstance(value, float) and float(parsed) != value:
        raise ValueError("max_candidates must be an integer")
    if parsed <= 0:
        raise ValueError("max_candidates must be positive")
    return parsed


def _baseline_fraction(params: Mapping[str, Any]) -> float:
    value = params.get("baseline_fraction", DEFAULT_BASELINE_FRACTION)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("baseline_fraction must be a number") from None
    if not 0 <= parsed <= 1:
        raise ValueError("baseline_fraction must be between 0 and 1")
    return parsed


def _reading_direction(params: Mapping[str, Any]) -> str:
    direction = params.get("reading_direction", "ltr")
    if direction not in ("ltr", "rtl"):
        raise ValueError('reading_direction must be "ltr" or "rtl"')
    return direction


def run_ppocr_det_segment(
    image_bytes: bytes,
    *,
    model_path: Path,
    artifact_sha256: str | None = None,
    params: dict[str, Any] | None = None,
) -> SegmentRunResponse:
    """Run PP-OCRv6 detection and return the segment contract in reading order."""

    # Resolved once here so a missing, mis-suffixed or tampered artifact fails
    # before the resize is paid for; the session loader re-resolves from a
    # memoized digest, which costs a ``stat``.
    _resolve_ppocr_det_artifact(model_path, artifact_sha256)
    resolved = params or {}
    limit = _limit_side_len(resolved)
    thresh = _positive_float_param(resolved, "thresh", DEFAULT_THRESH)
    box_thresh = _positive_float_param(resolved, "box_thresh", DEFAULT_BOX_THRESH)
    unclip_ratio = _positive_float_param(resolved, "unclip_ratio", DEFAULT_UNCLIP_RATIO)
    max_candidates = _max_candidates(resolved)
    fraction = _baseline_fraction(resolved)
    direction = _reading_direction(resolved)

    with open_image_bytes(image_bytes) as image:
        image = image.convert("RGB")
        width, height = image.size
        tensor, meta = preprocess_ppocr_det_image(image, limit_side_len=limit)
        _, _, resized_height, resized_width = tensor.shape

        handle = _resolve_ppocr_det_artifact(model_path, artifact_sha256)
        session, input_name, output_name = _load_ppocr_det_session(handle.path, handle.fingerprint)
        outputs = session.run([output_name], {input_name: np.ascontiguousarray(tensor)})
        prob = np.asarray(outputs[0], dtype=np.float32)
        if prob.shape != (1, 1, resized_height, resized_width):
            raise PPOCRDetUnavailableError(
                "PP-OCRv6 det ONNX graph must return [1, 1, H, W] matching the input"
            )
        quads = detect_lines(
            prob[0, 0],
            orig_width=width,
            orig_height=height,
            ratio_h=meta.ratio_h,
            ratio_w=meta.ratio_w,
            thresh=thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            max_candidates=max_candidates,
        )
        return build_ppocr_det_response(
            width,
            height,
            quads,
            baseline_fraction=fraction,
            reading_direction=direction,
        )


__all__ = [
    "PPOCRDetUnavailableError",
    "run_ppocr_det_segment",
]
