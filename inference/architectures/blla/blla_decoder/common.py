"""Common heatmap validation helpers."""

from __future__ import annotations

import numpy as np


def as_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Validate and truncate BLLA output to its four channels."""

    values = np.asarray(heatmaps, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 4:
        raise ValueError("BLLA heatmaps must have shape (4, height, width)")
    return values[:4]
