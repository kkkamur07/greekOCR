"""Torch-free BLLA inference through ONNX Runtime."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from src.hf.resolve.artifacts import verify_artifact_sha256

from inference.architectures.blla.blla_preprocessing import preprocess_blla_image_numpy
from inference.architectures.blla.blla_runtime import build_blla_segment_response
from inference.contracts.segment import SegmentRunResponse


class BLLAOnnxUnavailableError(RuntimeError):
    """Raised when the ONNX BLLA runtime cannot be used."""


def _validate_model_path(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"BLLA ONNX model not found: {model_path}")
    if model_path.suffix != ".onnx":
        raise BLLAOnnxUnavailableError(
            f"ONNX BLLA runtime requires an .onnx model: {model_path}"
        )


def _file_fingerprint(path: Path) -> tuple[int, int]:
    """Cache-key component so replaced artifact files are reloaded."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _load_blla_onnx_session(
    model_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[Any, str]:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        metadata = session.get_modelmeta().custom_metadata_map
        if metadata.get("format") != "blla-onnx-v1":
            raise BLLAOnnxUnavailableError("unsupported BLLA ONNX model format")
        if metadata.get("input_layout") != "NCHW":
            raise BLLAOnnxUnavailableError("unsupported BLLA ONNX input layout")
        if metadata.get("input_height") != "1800" or metadata.get("input_channels") != "3":
            raise BLLAOnnxUnavailableError("unsupported BLLA ONNX input dimensions")
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise BLLAOnnxUnavailableError("BLLA ONNX graph must have one input and output")
        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        if len(input_shape) != 4 or len(output_shape) != 4:
            raise BLLAOnnxUnavailableError("BLLA ONNX graph must use 4D tensors")
        return session, inputs[0].name
    except BLLAOnnxUnavailableError:
        raise
    except ImportError as error:
        raise BLLAOnnxUnavailableError(
            "onnxruntime is required for the BLLA ONNX runtime"
        ) from error
    except Exception as error:
        raise BLLAOnnxUnavailableError("unable to load BLLA ONNX model") from error


def run_blla_onnx_logits(
    inputs: np.ndarray,
    *,
    model_path: Path,
    artifact_sha256: str | None = None,
) -> np.ndarray:
    """Run the ONNX graph on one float32 NCHW NumPy input."""

    _validate_model_path(model_path)
    if artifact_sha256:
        verify_artifact_sha256(model_path, artifact_sha256)
    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("BLLA ONNX input must have shape (1, 3, 1800, width)")
    if values.shape[1] != 3 or values.shape[2] != 1800 or values.shape[3] <= 0:
        raise ValueError("BLLA ONNX input must have shape (1, 3, 1800, width)")

    session, input_name = _load_blla_onnx_session(str(model_path), _file_fingerprint(model_path))
    outputs = session.run(None, {input_name: np.ascontiguousarray(values)})
    logits = np.asarray(outputs[0], dtype=np.float32)
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 4:
        raise BLLAOnnxUnavailableError("BLLA ONNX graph returned invalid logits")
    return logits


def run_blla_onnx_segment(
    image_bytes: bytes,
    *,
    model_path: Path,
    artifact_sha256: str | None = None,
    params: dict[str, Any] | None = None,
) -> SegmentRunResponse:
    """Run Torch-free BLLA and return the legacy-compatible segment contract."""

    _validate_model_path(model_path)
    if artifact_sha256:
        verify_artifact_sha256(model_path, artifact_sha256)

    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        prepared = preprocess_blla_image_numpy(image)
        logits = run_blla_onnx_logits(
            prepared.array[None, ...],
            model_path=model_path,
        )[0]
        return build_blla_segment_response(
            image,
            logits,
            prepared,
            params=params,
            torch_free=True,
        )


run_blla_segment_onnx = run_blla_onnx_segment


__all__ = [
    "BLLAOnnxUnavailableError",
    "run_blla_onnx_logits",
    "run_blla_onnx_segment",
    "run_blla_segment_onnx",
]
