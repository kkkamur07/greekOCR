"""Grayscale loading matching the training loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_line_image_grayscale(image_path: Path) -> np.ndarray:
    """Load a line image as grayscale, matching what the model is trained on.

    The training loader (``src/models/calamari/data.py::_load_line_image``) and
    the training crop writer (``src/preprocessing_data/syriac/xml_to_data.py``)
    both go through ``Image.convert("L")``; so does the serving path in
    ``pipeline.py``. A previous implementation dispatched on channel *count*
    from the raw PIL mode, which agrees with ``convert("L")`` only for
    RGB/RGBA/L sources. For a palette PNG it handed the model palette indices as
    if they were luminance (measured: 98.6% of pixels differ, up to 231 levels);
    CMYK and I;16 were similarly wrong, and LA raised outright.
    """
    with Image.open(image_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)
