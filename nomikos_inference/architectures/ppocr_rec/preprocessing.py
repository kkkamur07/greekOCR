"""PP-OCR recognition line preprocessing for the ONNX inference runtime.

Serving must reproduce kraken's recognition input recipe exactly. The model was
trained through kraken's ``ImageInputTransforms`` with
``(batch=1, height=96, width=0, channels=3, pad=(16, 0))``, which composes, in
order:

1. ``pil_to_mode('RGB')``,
2. ``pil_fixed_resize(scale=(96, 0))``: aspect-preserving resize to the line
   height with ``Resampling.LANCZOS`` and the target width truncated, not
   rounded (``ow = int(w * oh / h)`` in
   ``kraken/lib/functional_im_transforms.py``),
3. ``v2.Pad((16, 0), fill=255)``: white columns on the left AND right, 16 px
   each (a 2-tuple pads left/right and top/bottom respectively),
4. ``PILToTensor`` plus ``ToDtype(float32, scale=True)``: ``uint8`` to
   ``float32`` scaled to [0, 1] by *multiplying* with ``1/255`` (torchvision
   scales int to float with ``to(dtype).mul_(1.0 / max)``, and multiply
   rounds differently from divide on some values, so ``x / 255`` is not
   bitwise identical),
5. ``tensor_invert``: ``1 - x`` (the white padding guarantees the tensor
   maximum is 1.0, so kraken's ``max - x`` and ``1 - x`` agree),
6. a no-op permute, then batching to ``[1, 3, 96, W]``.

This module is that recipe with PIL and numpy only: the server is torch-free,
so torchvision is unavailable. The intermediate PIL steps are exposed as
functions so tests can compare each stage against kraken's own transforms.

One rule beyond kraken: lines whose padded width is below 16 px are extended
on the right with ``pad_fill`` up to 16 before inversion. The graph was
verified from width 16 up, and a narrower input would otherwise produce a time
dimension the backbone was never checked at.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

#: Minimum preprocessed width the graph is verified at. Narrower lines are
#: extended on the right before inversion.
MIN_PREPROCESSED_WIDTH = 16


def open_line_image(image_bytes: bytes) -> Image.Image:
    """Decode line bytes to an RGB PIL image (kraken's ``pil_to_mode``)."""
    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def fixed_resize_to_height(image: Image.Image, line_height: int) -> Image.Image:
    """Aspect-preserving resize to ``line_height`` (kraken's ``pil_fixed_resize``).

    The target width is truncated exactly as kraken truncates it
    (``ow = int(w * oh / h)``), and the resample filter is ``LANCZOS``, the
    filter ``_fixed_resize`` passes explicitly.
    """
    if line_height <= 0:
        raise ValueError("line_height must be positive")
    width, height = image.size
    if height <= 0:
        raise ValueError("line image has no height")
    target_width = int(width * line_height / height)
    return image.resize((target_width, line_height), Image.Resampling.LANCZOS)


def pad_line_sides(image: Image.Image, pad: int, pad_fill: int) -> Image.Image:
    """Add ``pad`` fill columns on the left and right (kraken's ``v2.Pad``)."""
    if pad < 0:
        raise ValueError("pad must be non-negative")
    if not 0 <= pad_fill <= 255:
        raise ValueError("pad_fill must be a uint8 value")
    if pad == 0:
        return image
    padded = Image.new("RGB", (image.width + 2 * pad, image.height), (pad_fill,) * 3)
    padded.paste(image, (pad, 0))
    return padded


def ensure_minimum_width(image: Image.Image, pad_fill: int) -> Image.Image:
    """Extend narrow lines on the right up to ``MIN_PREPROCESSED_WIDTH``.

    Only the right side grows: the left padding is part of the recipe the
    model was trained on, while the right edge past the text carries no
    signal the backbone was checked without.
    """
    if image.width >= MIN_PREPROCESSED_WIDTH:
        return image
    extended = Image.new("RGB", (MIN_PREPROCESSED_WIDTH, image.height), (pad_fill,) * 3)
    extended.paste(image, (0, 0))
    return extended


def preprocess_line_image_bytes_to_ppocr_rec_tensor(
    image_bytes: bytes,
    *,
    line_height: int,
    pad: int,
    pad_fill: int,
) -> np.ndarray:
    """Return the PP-OCR model input as float32 ``[1, 3, H, W]``.

    The kraken recipe applied to the line crop: RGB, fixed-height resize
    preserving aspect ratio, white padding on the left and right, scaled to
    [0, 1], then inverted (``1 - x``).
    """
    image = open_line_image(image_bytes)
    if image.width == 0 or image.height == 0:
        raise ValueError("line image is empty")
    image = fixed_resize_to_height(image, line_height)
    image = pad_line_sides(image, pad, pad_fill)
    image = ensure_minimum_width(image, pad_fill)
    # Multiply, not divide: torchvision's ``ToDtype`` scales int to float as
    # ``image.to(dtype).mul_(1.0 / 255)``, and ``x * float32(1/255)`` rounds
    # differently from ``x / 255`` on some inputs (1 ulp). The reciprocal is
    # the float64 ``1.0 / 255`` narrowed to float32, matching torch's scalar.
    pixels = np.asarray(image, dtype=np.uint8).astype(np.float32)
    scaled = pixels * np.float32(1.0 / 255.0)
    inverted = 1.0 - scaled
    return np.ascontiguousarray(inverted.transpose(2, 0, 1))[None]


__all__ = [
    "MIN_PREPROCESSED_WIDTH",
    "ensure_minimum_width",
    "fixed_resize_to_height",
    "open_line_image",
    "pad_line_sides",
    "preprocess_line_image_bytes_to_ppocr_rec_tensor",
]
