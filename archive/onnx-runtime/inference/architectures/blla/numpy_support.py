"""Torch-free BLLA preprocessing and decoding helpers.

RETIRED. See ``archive/onnx-runtime/README.md`` and ADR 0004.

These were the pieces of ``inference/architectures/blla/`` that existed only so
the ONNX Runtime adapter could run a page without importing Torch. They were
removed from the live tree when PyTorch became the runtime and are collected
here so ``blla/onnx.py`` beside them stays readable.

Original locations:

* ``BLLANumpyInput`` and ``preprocess_blla_image_numpy``
  - ``inference/architectures/blla/blla_preprocessing.py``
* ``resize_heatmaps_nearest``
  - ``inference/architectures/blla/blla_decoder/common.py``
* ``sigmoid_or_passthrough``
  - inlined in ``inference/architectures/blla/blla_decoder/__init__.py`` behind
    the ``torch_free`` flag, alongside the Torch branch that survives today

``build_blla_segment_response`` and ``decode_blla_heatmaps`` both took a
``torch_free: bool`` that selected between these implementations and their
Torch equivalents. The parity harness pinned that the two decoders produced
identical baselines and polygons from identical logits, so any ONNX drift was
attributable to the graph rather than to this code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from inference.architectures.blla.blla_decoder.common import as_heatmaps
from inference.architectures.blla.blla_preprocessing import _scaled_blla_width


@dataclass(frozen=True)
class BLLANumpyInput:
    """The Torch-free model input and image-space representation."""

    array: np.ndarray
    scaled_gray: np.ndarray
    scale_xy: tuple[float, float]


def preprocess_blla_image_numpy(
    image: Image.Image,
    *,
    input_height: int = 1800,
) -> BLLANumpyInput:
    """Prepare BLLA input without importing or constructing Torch tensors."""

    if input_height <= 0:
        raise ValueError("input_height must be positive")
    rgb = image.convert("RGB")
    source_width, source_height = rgb.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("BLLA input image must not be empty")

    scaled_width = _scaled_blla_width(source_width, source_height, input_height)
    scaled = rgb.resize((scaled_width, input_height), Image.Resampling.LANCZOS)
    scaled_array = np.asarray(scaled, dtype=np.uint8)
    scaled_gray = np.asarray(scaled.convert("L"), dtype=np.uint8)
    array = np.transpose(scaled_array, (2, 0, 1)).astype(np.float32, copy=True)
    array /= np.float32(255.0)
    array = np.float32(array.max()) - array
    return BLLANumpyInput(
        array=array,
        scaled_gray=scaled_gray,
        scale_xy=(source_width / scaled_width, source_height / input_height),
    )


def resize_heatmaps_nearest(
    heatmaps: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Resize channels with the nearest-neighbour rule used by Torch.

    BLLA's reference decoder uses ``torch.nn.functional.interpolate`` without
    a mode, which is nearest-neighbour interpolation for a 4D tensor. Keeping
    this small operation in NumPy removed Torch from the decoder's runtime
    dependency without changing the reference operation.
    """

    if height <= 0 or width <= 0:
        raise ValueError("heatmap output size must be positive")
    values = as_heatmaps(heatmaps)
    source_height, source_width = values.shape[1:]
    y_indices = np.minimum(
        np.arange(height, dtype=np.int64) * source_height // height,
        source_height - 1,
    )
    x_indices = np.minimum(
        np.arange(width, dtype=np.int64) * source_width // width,
        source_width - 1,
    )
    return values[:, y_indices[:, None], x_indices[None, :]]


def sigmoid_or_passthrough(resized: np.ndarray, *, raw_logits: bool) -> np.ndarray:
    """The NumPy half of the decoder's ``torch_free`` branch."""

    if not raw_logits:
        return resized
    return np.reciprocal(np.add(1.0, np.exp(-resized), dtype=np.float32))
