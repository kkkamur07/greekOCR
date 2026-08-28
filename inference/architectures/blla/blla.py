"""BLLA segmentation on the ONNX Runtime CPU runtime (ADR 0006).

Uses the plain names (``BLLAUnavailableError``, ``run_blla_segment``) since
this is the only BLLA adapter here. ``resolve_artifact`` verifies the
**artifact SHA-256** before the file is opened.

The **Hub artifact** is ``blla.onnx``. Unlike a pickled ``.pt`` it carries no
executable payload, but the digest is still checked: onnxruntime parses the
protobuf in C++, and an artifact that does not match its pin is a broken
deployment however it fails.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from inference.admission import open_image_bytes
from inference.architectures.artifact import ArtifactHandle, resolve_artifact
from inference.architectures.blla.blla_preprocessing import preprocess_blla_image
from inference.architectures.blla.blla_runtime import build_blla_segment_response
from inference.contracts.segment import SegmentRunResponse

BLLA_ARTIFACT_SUFFIXES = frozenset({".onnx"})

# The graph has a fixed input height and a fixed channel count; the width is the
# only free axis. Both are asserted against the artifact's own metadata rather
# than assumed, because a re-export at a different height would otherwise
# produce silently mis-scaled baselines instead of an error.
BLLA_INPUT_HEIGHT = 1800
BLLA_INPUT_CHANNELS = 3


class BLLAUnavailableError(RuntimeError):
    """Raised when a BLLA runtime artifact cannot be used."""


def _resolve_blla_artifact(
    model_path: Path,
    artifact_sha256: str | None = None,
) -> ArtifactHandle:
    return resolve_artifact(
        model_path,
        label="BLLA model",
        allowed_suffixes=BLLA_ARTIFACT_SUFFIXES,
        unusable_error=BLLAUnavailableError,
        unusable_message=f"BLLA runtime requires an .onnx model: {model_path}",
        artifact_sha256=artifact_sha256,
    )


@lru_cache(maxsize=4)
def _load_blla_session(
    model_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[Any, str]:
    """Open a session and check the graph is the one this decoder can read.

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
        metadata = session.get_modelmeta().custom_metadata_map
        if metadata.get("format") != "blla-onnx-v1":
            raise BLLAUnavailableError("unsupported BLLA ONNX model format")
        if metadata.get("input_layout") != "NCHW":
            raise BLLAUnavailableError("unsupported BLLA ONNX input layout")
        if metadata.get("input_height") != str(BLLA_INPUT_HEIGHT) or metadata.get(
            "input_channels"
        ) != str(BLLA_INPUT_CHANNELS):
            raise BLLAUnavailableError("unsupported BLLA ONNX input dimensions")
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise BLLAUnavailableError("BLLA ONNX graph must have one input and output")
        if len(inputs[0].shape) != 4 or len(outputs[0].shape) != 4:
            raise BLLAUnavailableError("BLLA ONNX graph must use 4D tensors")
        return session, inputs[0].name
    except BLLAUnavailableError:
        raise
    except ImportError as error:
        raise BLLAUnavailableError("onnxruntime is required for the BLLA runtime") from error
    except Exception as error:
        raise BLLAUnavailableError("unable to load BLLA ONNX model") from error


def run_blla_logits(
    inputs: np.ndarray,
    *,
    model_path: Path,
    artifact_sha256: str | None = None,
) -> np.ndarray:
    """Run the graph on one float32 NCHW NumPy input."""

    handle = _resolve_blla_artifact(model_path, artifact_sha256)
    values = np.asarray(inputs, dtype=np.float32)
    expected = f"BLLA input must have shape (1, {BLLA_INPUT_CHANNELS}, {BLLA_INPUT_HEIGHT}, width)"
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError(expected)
    if (
        values.shape[1] != BLLA_INPUT_CHANNELS
        or values.shape[2] != BLLA_INPUT_HEIGHT
        or values.shape[3] <= 0
    ):
        raise ValueError(expected)

    session, input_name = _load_blla_session(handle.path, handle.fingerprint)
    outputs = session.run(None, {input_name: np.ascontiguousarray(values)})
    logits = np.asarray(outputs[0], dtype=np.float32)
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 4:
        raise BLLAUnavailableError("BLLA ONNX graph returned invalid logits")
    return logits


def run_blla_segment(
    image_bytes: bytes,
    *,
    model_path: Path,
    artifact_sha256: str | None = None,
    params: dict[str, Any] | None = None,
) -> SegmentRunResponse:
    """Run BLLA and return the legacy-compatible segment contract."""

    # Resolved once here so a missing, mis-suffixed or tampered artifact fails
    # before a full-page Lanczos resize is paid for; ``run_blla_logits``
    # re-resolves from a memoized digest, which costs a ``stat``.
    _resolve_blla_artifact(model_path, artifact_sha256)

    with open_image_bytes(image_bytes) as image:
        image = image.convert("RGB")
        prepared = preprocess_blla_image(image, input_height=BLLA_INPUT_HEIGHT)
        logits = run_blla_logits(
            prepared.array[None, ...],
            model_path=model_path,
            artifact_sha256=artifact_sha256,
        )[0]
        return build_blla_segment_response(image, logits, prepared, params=params)


__all__ = [
    "BLLAUnavailableError",
    "run_blla_logits",
    "run_blla_segment",
]
