"""Final Calamari line image normalization before model inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from inference.architectures.calamari.preprocessing.conversion import to_float32, to_uint8


def final_prepare_line_image(
    image: np.ndarray,
    *,
    normalize: bool = True,
    invert: bool = True,
    transpose: bool = True,
    pad: int = 16,
    pad_value: int = 0,
    meta: dict[str, Any] | None = None,
) -> np.ndarray:
    """Mirror Calamari's FinalPreparation processor."""
    data = to_float32(image)
    if data.ndim != 3:
        data = np.expand_dims(data, axis=-1)

    channels = data.shape[-1]
    if data.size > 0:
        if normalize:
            amax = np.amax(data)
            if amax > 0:
                data = data * 1.0 / amax
        if invert:
            data = np.amax(data) - data

    if transpose:
        data = np.swapaxes(data, 1, 0)

    if pad > 0:
        if transpose:
            width = data.shape[1]
            data = np.vstack(
                [
                    np.full((pad, width, channels), pad_value),
                    data,
                    np.full((pad, width, channels), pad_value),
                ]
            )
            if meta is not None:
                meta["padded_width"] = data.shape[0]
        else:
            width = data.shape[0]
            data = np.hstack(
                [
                    np.full((width, pad, channels), pad_value),
                    data,
                    np.full((width, pad, channels), pad_value),
                ]
            )
            if meta is not None:
                meta["padded_width"] = data.shape[1]

    data = to_uint8(data)
    if channels == 1:
        data = np.squeeze(data, axis=-1)
    return data
