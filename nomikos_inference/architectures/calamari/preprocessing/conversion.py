"""Image dtype and grayscale conversion matching Calamari utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_line_image_grayscale(image_path: Path) -> np.ndarray:
    """Load a line image as grayscale, matching what the served model is fed.

    This must stay identical to the serving path in ``pipeline.py``, which does
    ``image.convert("L")``. The previous implementation dispatched on channel
    *count* from the raw PIL mode, which agrees with ``convert("L")`` only for
    RGB/RGBA/L sources. For a palette PNG it handed the model palette indices as
    if they were luminance (measured: 98.6% of pixels differ, up to 231 levels);
    CMYK and I;16 were similarly wrong, and LA raised outright. Training on that
    distribution while serving another is a skew no test was watching for.

    ``convert("L")`` and OpenCV's ``COLOR_RGB2GRAY`` use the same ITU-R 601-2
    coefficients and differ by at most 1 LSB on true RGB input, so this does not
    disturb the RGB corpus - it fixes the modes that were never handled.
    """
    with Image.open(image_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


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
