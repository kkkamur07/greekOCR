"""Input preprocessing for the BLLA graph.

NumPy only (ADR 0006: no Torch adapter). The array it produces is what
onnxruntime binds directly. Arithmetic: divide by 255, invert around the
array maximum, matching what the model was traced on;
``tests/inference/unit/test_onnx_parity.py`` verifies this against the real
weights.
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
#
# The bound also limits numerics: ONNX's agreement with the Torch oracle
# decays with scaled width. Measured on the ``segment_page.jpeg`` fixture
# tiled to each width, ONNX vs Torch on raw logits:
#
#   width   rms |d|   p99.9 |d|   max |d|   logits crossing sigmoid 0.5
#    2471   1.7e-05     1.4e-04   1.5e-03   0        <- a real page, ADR 0006
#    3600   2.2e-05     1.5e-04   1.1e-03   0
#    5400   3.6e-05     2.5e-04   2.3e-03   0
#    7200   7.5e-05     4.7e-04   1.2e-02   1
#    9000   9.1e-05     6.1e-04   2.1e-02   1
#   14400   1.9e-04     1.9e-03   2.4e-02   3
#
# Judge by the RMS column: the max is a noisy extreme-value statistic, and the
# flip count just tracks content. RMS scales roughly linearly with width, and a
# bound of 3 keeps it within about 3x the page ADR 0006 validated. Sigmoid
# flips (the decoder is discontinuous at 0.5, so a handful of pixels can
# restructure a line polygon) become routine from width 7200 up.
#
# Cost: a source wider than 3:1 is squeezed horizontally, up to 2.67x for a
# true panorama. A single leaf (~0.7:1) or two-page spread (~2.5:1) is
# unaffected; the stitched scroll is what pays.
#
# The growth traces to ``Gn_13``, the one GroupNorm with channels-per-group=1,
# whose reduction is dominated by width. Re-exported with float32 staged
# reduction vs. float64-accumulated moments:
#
#   feature width   staged rel |d|   float64 rel |d|
#     618 (page  2471)   2.16e-06        2.14e-06
#    1350 (page  5400)   5.23e-06        5.21e-06
#    2250 (page  9000)   1.41e-05        1.41e-05
#    3600 (page 14400)   4.63e-05        4.63e-05
#
# The two agree to six figures, so the drift isn't ONNX Runtime's reduction,
# it's Torch's own float32 ``group_norm`` at that size. No change to
# ``src/model/inference_export/blla/export.py`` can close it; only a shorter
# width can.
MAX_WIDTH_TO_HEIGHT_RATIO = 3


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
