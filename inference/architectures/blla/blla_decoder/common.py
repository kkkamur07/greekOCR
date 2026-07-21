"""Common heatmap validation helpers."""

from __future__ import annotations

import numpy as np


def as_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Validate and truncate BLLA output to its four channels."""

    values = np.asarray(heatmaps, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 4:
        raise ValueError("BLLA heatmaps must have shape (4, height, width)")
    return values[:4]


def resize_heatmaps_nearest(
    heatmaps: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Resize channels with the nearest-neighbour rule used by Torch.

    BLLA's reference decoder uses ``torch.nn.functional.interpolate`` without
    a mode, which is nearest-neighbour interpolation for a 4D tensor. Keeping
    this small operation in NumPy removes Torch from the decoder's runtime
    dependency without changing the reference operation.
    """

    if height <= 0 or width <= 0:
        raise ValueError("heatmap output size must be positive")
    values = as_heatmaps(heatmaps)
    source_height, source_width = values.shape[1:]
    y_indices = np.minimum(
        np.arange(height, dtype=np.int64) * source_height // height,
        source_height - 1,
    )
    x_indices = np.minimum(
        np.arange(width, dtype=np.int64) * source_width // width,
        source_width - 1,
    )
    return values[:, y_indices[:, None], x_indices[None, :]]
