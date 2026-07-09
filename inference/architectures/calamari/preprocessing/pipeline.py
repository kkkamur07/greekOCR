"""Public Calamari preprocessing pipeline for PyTorch inference."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from inference.architectures.calamari.preprocessing.conversion import load_line_image_grayscale
from inference.architectures.calamari.preprocessing.final import final_prepare_line_image
from inference.architectures.calamari.preprocessing.geometry import center_normalize


def preprocess_line_image_to_calamari_tensor(
    image_path: Path,
    *,
    line_height: int = 48,
    pad: int = 16,
    pad_value: int = 0,
) -> np.ndarray:
    """Return Calamari model input as ``batch x time x height x channel`` uint8."""
    image = load_line_image_grayscale(image_path)
    return preprocess_line_array_to_calamari_tensor(
        image,
        line_height=line_height,
        pad=pad,
        pad_value=pad_value,
    )


def preprocess_line_image_bytes_to_calamari_tensor(
    image_bytes: bytes,
    *,
    line_height: int = 48,
    pad: int = 16,
    pad_value: int = 0,
) -> np.ndarray:
    """Return Calamari model input for encoded image bytes."""
    with Image.open(BytesIO(image_bytes)) as image:
        image_array = np.asarray(image.convert("L"), dtype=np.uint8)
    return preprocess_line_array_to_calamari_tensor(
        image_array,
        line_height=line_height,
        pad=pad,
        pad_value=pad_value,
    )


def preprocess_line_array_to_calamari_tensor(
    image: np.ndarray,
    *,
    line_height: int = 48,
    pad: int = 16,
    pad_value: int = 0,
) -> np.ndarray:
    """Return Calamari model input for a grayscale uint8 line array."""
    meta: dict[str, Any] = {}
    image = center_normalize(image, line_height=line_height, meta=meta)
    image = final_prepare_line_image(
        image,
        normalize=True,
        invert=True,
        transpose=True,
        pad=pad,
        pad_value=pad_value,
        meta=meta,
    )
    if image.ndim == 2:
        image = image[:, :, None]
    return image[None, :, :, :]
