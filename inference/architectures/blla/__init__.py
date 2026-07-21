"""BLLA segmentation adapters for native Torch and ONNX Runtime."""

from __future__ import annotations

from typing import Any


def run_blla_segment(*args: Any, **kwargs: Any) -> Any:
    """Load the native adapter lazily so ONNX-only imports stay Torch-free."""

    from inference.architectures.blla.blla import run_blla_segment as run_native

    return run_native(*args, **kwargs)


def run_blla_onnx_segment(*args: Any, **kwargs: Any) -> Any:
    """Load the Torch-free ONNX adapter lazily."""

    from inference.architectures.blla.onnx import (
        run_blla_onnx_segment as run_onnx,
    )

    return run_onnx(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "BLLAUnavailableError":
        from inference.architectures.blla.blla import BLLAUnavailableError

        return BLLAUnavailableError
    if name == "BLLAOnnxUnavailableError":
        from inference.architectures.blla.onnx import BLLAOnnxUnavailableError

        return BLLAOnnxUnavailableError
    raise AttributeError(name)

__all__ = [
    "BLLAUnavailableError",
    "BLLAOnnxUnavailableError",
    "run_blla_segment",
    "run_blla_onnx_segment",
]
