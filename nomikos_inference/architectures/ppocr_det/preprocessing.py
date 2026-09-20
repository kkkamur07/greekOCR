"""Input preprocessing for the PP-OCRv6 detection graph.

Reproduces PaddleX 3.7.0 exactly, so the ONNX graph sees the pixels it was
trained on:

* resize: ``paddlex/inference/models/text_detection/processors.py``,
  ``DetResizeForTest.resize_image_type0`` with ``limit_type="max"`` (scale
  down only when the longer side exceeds the limit, never upscale), each side
  rounded to a multiple of 32 with a floor of 32, ``cv2.resize`` with its
  default bilinear interpolation;
* channel order: ``DecodeImage`` in this model's ``inference.yml`` sets
  ``img_mode: BGR``, and PaddleX's ``ReadImage`` leaves an OpenCV BGR array
  untouched, so the tensor is BGR;
* normalisation: ``NormalizeImage`` (same file) with ``order: hwc`` applies
  ``scale=1/255``, ``mean=[0.485, 0.456, 0.406]`` and
  ``std=[0.229, 0.224, 0.225]`` per stored channel, which for a BGR image
  means channel 0 (blue) is normalised with 0.485/0.229;
* layout: ``ToCHWImage`` then ``ToBatch`` in
  ``paddlex/inference/models/common/vision/processors.py`` (CHW, float32,
  batch axis).

The one PaddleX rule not reproduced is the ``max_side_limit`` recap inside
``resize_image_type0``: with the admitted ``limit_side_len`` range
(320 to 4000) the 32-rounded sides can never exceed the PaddleX cap of 4000,
so the recap is unreachable. Tiny images (height plus width below 64), which
PaddleX zero pads to at least 32 a side with the image top left before the
multiple-of-32 logic, are reproduced below: the ratios are over the padded
dims exactly as PaddleX computes them, while the box mapping back to source
pixels divides by the resized dims against the original size, which is what
PaddleX's own ``dest`` over ``width`` scale does.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# ImageNet statistics in the stored (BGR) channel order, exactly as
# ``NormalizeImage`` applies them: position 0 normalises whatever is in
# channel 0, which PaddleX's BGR decode leaves as blue.
PPOCR_DET_MEAN = (0.485, 0.456, 0.406)
PPOCR_DET_STD = (0.229, 0.224, 0.225)
PPOCR_DET_SCALE = 1.0 / 255.0


@dataclass(frozen=True)
class PPOCRDetMeta:
    """Sizes and ratios needed to map detections back to the source page."""

    orig_width: int
    orig_height: int
    ratio_h: float
    ratio_w: float


def _resized_dims(width: int, height: int, limit_side_len: int) -> tuple[int, int]:
    longer = max(width, height)
    ratio = limit_side_len / longer if longer > limit_side_len else 1.0
    resized_w = int(width * ratio)
    resized_h = int(height * ratio)
    # Nearest multiple of 32 with a floor of 32, as in ``resize_image_type0``.
    # This also snaps images that needed no limit scaling, so a page below the
    # limit still grows slightly (for example 600 px becomes 608).
    resized_w = max(int(round(resized_w / 32)) * 32, 32)
    resized_h = max(int(round(resized_h / 32)) * 32, 32)
    return resized_w, resized_h


def preprocess_ppocr_det_image(
    image: Image.Image,
    *,
    limit_side_len: int = 1920,
) -> tuple[np.ndarray, PPOCRDetMeta]:
    """Resize, normalise and batch one page the way PaddleX does.

    Returns the ``[1, 3, H, W]`` float32 model input and the meta that maps
    resized coordinates back to source pixels.
    """

    if limit_side_len <= 0:
        raise ValueError("limit_side_len must be positive")
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise ValueError("PP-OCRv6 det input image must not be empty")

    # ``np.asarray`` of a PIL RGB image is HWC RGB; reversing the last axis is
    # the BGR order ``cv2.imread`` would have produced in the PaddleX pipeline.
    bgr = np.asarray(rgb)[:, :, ::-1]
    # PaddleX ``DetResizeForTest.resize`` zero pads images with height plus
    # width below 64 to at least 32 a side, image top left, before the
    # multiple-of-32 resize runs over the padded dims.
    padded_height, padded_width = height, width
    if height + width < 64:
        padded_height, padded_width = max(32, height), max(32, width)
        canvas = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
        canvas[:height, :width, :] = bgr
        bgr = canvas
    resized_w, resized_h = _resized_dims(padded_width, padded_height, limit_side_len)
    # Default interpolation is bilinear, matching PaddleX's bare
    # ``cv2.resize(img, (w, h))`` call.
    resized = cv2.resize(bgr, (resized_w, resized_h))

    normalised = resized.astype(np.float32) * np.float32(PPOCR_DET_SCALE)
    for channel in range(3):
        normalised[:, :, channel] = (normalised[:, :, channel] - PPOCR_DET_MEAN[channel]) / (
            PPOCR_DET_STD[channel]
        )
    chw = np.transpose(normalised, (2, 0, 1))
    tensor = np.ascontiguousarray(chw[None, ...], dtype=np.float32)
    meta = PPOCRDetMeta(
        orig_width=width,
        orig_height=height,
        ratio_h=resized_h / float(padded_height),
        ratio_w=resized_w / float(padded_width),
    )
    return tensor, meta


__all__ = [
    "PPOCRDetMeta",
    "preprocess_ppocr_det_image",
]
