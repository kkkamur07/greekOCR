"""Image dtype and grayscale conversion matching Calamari utilities."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image


def load_line_image_grayscale(image_path: Path) -> np.ndarray:
    """Load an image using Calamari's RGB-to-gray convention."""
    with Image.open(image_path) as image:
        data = to_uint8(np.asarray(image))

    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[-1] == 1:
        return data[:, :, 0]
    if data.ndim == 3 and data.shape[-1] == 3:
        return cv.cvtColor(data, cv.COLOR_RGB2GRAY)
    if data.ndim == 3 and data.shape[-1] == 4:
        return cv.cvtColor(data, cv.COLOR_RGBA2GRAY)
    raise ValueError(f"unsupported image shape for grayscale conversion: {data.shape}")


def to_uint8(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.dtype("uint8"):
        return data
    if data.dtype == np.dtype("int8"):
        return (data.astype("int16") + 128).astype("uint8")
    if data.dtype == np.dtype("uint16"):
        return (data / 256).astype("uint8")
    if data.dtype == np.dtype("int16"):
        return ((data / 128).astype("int16") + 128).astype("uint8")
    if data.dtype in [np.dtype("f"), np.dtype("float32"), np.dtype("float64")]:
        return (data * 255).astype("uint8")
    if data.dtype == bool:
        return data.astype("uint8") * 255
    raise ValueError(f"unknown image type: {data.dtype}")


def to_float32(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.dtype("uint8"):
        return data.astype("float32") / 255
    if data.dtype == np.dtype("int8"):
        return (data.astype("int16") + 128).astype("float32") / 255
    if data.dtype == np.dtype("uint16"):
        return data.astype("float32") / 65535
    if data.dtype == np.dtype("int16"):
        return (data.astype("float32") + 32768) / 65535
    if data.dtype in [np.dtype("f"), np.dtype("float32"), np.dtype("float64")]:
        return data.astype("float32")
    if data.dtype == bool:
        return data.astype("float32")
    raise ValueError(f"unknown image type: {data.dtype}")
