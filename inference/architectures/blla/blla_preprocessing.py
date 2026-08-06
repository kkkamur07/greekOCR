"""Input preprocessing for the BLLA graph.

NumPy only. This is the ``preprocess_blla_image_numpy`` that ADR 0004 retired,
promoted back to being the one implementation: under ADR 0006 there is no Torch adapter
left to need a ``Tensor``, and the array it produces is what onnxruntime binds
directly.

The arithmetic is unchanged from the Torch version it replaces - divide by 255,
invert around the array maximum - so the pixels the graph sees are the pixels it
was traced on. ``tests/inference/unit/test_onnx_parity.py`` holds that claim to
the real weights rather than to this comment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# The model height is fixed (1800 by default) while the width scales with the
# source aspect ratio. Without a bound, an extreme panorama that still passes
# the pixel-count admission cap can balloon into a multi-gigabyte array, so
# clamp the scaled width to this multiple of the input height. Capped images
# lose horizontal resolution but still decode correctly because coordinates
# are mapped back through ``scale_xy``.
MAX_WIDTH_TO_HEIGHT_RATIO = 8


def _scaled_blla_width(source_width: int, source_height: int, input_height: int) -> int:
    proportional = int(source_width * input_height / source_height)
    return min(max(1, proportional), input_height * MAX_WIDTH_TO_HEIGHT_RATIO)


@dataclass(frozen=True)
class BLLAInput:
    """The model input and its image-space representation."""

    array: np.ndarray
    scaled_gray: np.ndarray
    scale_xy: tuple[float, float]


def preprocess_blla_image(
    image: Image.Image,
    *,
    input_height: int = 1800,
) -> BLLAInput:
    """Match the reference BLLA inference transforms.

    The shipped model has a fixed height and variable width. The reference
    pipeline converts to RGB, resizes proportionally with PIL Lanczos, scales
    to ``[0, 1]``, inverts around the array maximum, and leaves the channel
    order as RGB.
    """

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
    return BLLAInput(
        array=array,
        scaled_gray=scaled_gray,
        scale_xy=(source_width / scaled_width, source_height / input_height),
    )
