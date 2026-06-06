"""Binarize step — Kraken nonlinear binarization (nlbin)."""

import numpy as np
from PIL import Image


def binarize(image: np.ndarray) -> np.ndarray:
    """Binarize a line crop using Kraken's nlbin."""
    try:
        from kraken import binarization
    except ImportError as e:
        raise RuntimeError(
            "Kraken is required for binarization. Install with: pip install 'annote[kraken]'"
        ) from e

    pil = Image.fromarray(image)
    try:
        out = binarization.nlbin(pil)
    except Exception as e:
        raise RuntimeError(f"Kraken binarization failed: {e}") from e
    if out.mode != "RGB":
        out = out.convert("RGB")
    return np.array(out)
