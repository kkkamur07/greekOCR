"""Calamari center-normalization and dewarping geometry."""

from __future__ import annotations

from typing import Any

import cv2 as cv
import numpy as np


def center_normalize(
    image: np.ndarray,
    *,
    line_height: int,
    meta: dict[str, Any] | None = None,
    range_: int = 4,
    smoothness: float = 1.0,
    extra: float = 0.3,
) -> np.ndarray:
    """Mirror Calamari's CenterNormalizer image dewarping and height scaling."""
    if line_height <= 0:
        raise ValueError("line_height must be positive")

    image = image.astype(np.uint8)
    intermediate_height = int(line_height * 1.5)
    m1 = 1
    if intermediate_height < image.shape[0]:
        m1 = intermediate_height / image.shape[0]
        image = scale_to_height(image, intermediate_height)

    if image.size == 0:
        cval: int | list[int] = 1
    elif image.ndim == 2:
        cval = int(np.amax(image).item())
    else:
        x, y = np.unravel_index(np.argmax(np.mean(image, axis=2)), image.shape[:2])
        cval = image[x, y, :].tolist()

    dewarped = _dewarp(image, cval=cval, range_=range_, smoothness=smoothness, extra=extra)
    t = dewarped.shape[0] - image.shape[0]
    scaled = scale_to_height(dewarped, line_height)
    m2 = 1 if dewarped.size == 0 else scaled.shape[1] / dewarped.shape[1]

    if meta is not None:
        meta["center"] = (m1, m2, t)
    return scaled


def scale_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.dtype != np.dtype("uint8"):
        raise ValueError(f"scale_to_height expects uint8 data, got {image.dtype}")

    height, width = image.shape[:2]
    if height == target_height:
        return image
    if height == 0 or image.size == 0:
        return np.zeros(shape=(target_height, width) + image.shape[2:], dtype=image.dtype)

    scale = target_height * 1.0 / height
    target_width = max(round(scale * width), 1)
    interpolation = cv.INTER_AREA if scale <= 1 else cv.INTER_LINEAR
    return cv.resize(image, (target_width, target_height), interpolation=interpolation)


def _dewarp(
    image: np.ndarray,
    *,
    cval: int | list[int],
    range_: int,
    smoothness: float,
    extra: float,
) -> np.ndarray:
    if image.size == 0:
        return image

    if image.ndim > 2:
        if image.ndim != 3:
            raise ValueError(f"unsupported image rank for dewarp: {image.shape}")
        if image.shape[-1] == 1:
            temp = np.squeeze(image, axis=-1)
        elif image.shape[-1] == 3:
            temp = (cv.cvtColor(image, cv.COLOR_RGB2GRAY) / 255).astype(np.float32)
        else:
            temp = np.mean(image, axis=-1)
    else:
        temp = (image / 255).astype(np.float32)

    temp = np.amax(temp) - temp
    amax = np.amax(temp)
    if amax == 0:
        return (temp * 255).astype(np.uint8)

    inverted = temp * 1.0 / np.amax(temp)
    center, radius = _measure_centerline(
        inverted,
        range_=range_,
        smoothness=smoothness,
        extra=extra,
    )

    hpad = radius
    padded = cv.copyMakeBorder(image, hpad, hpad, 0, 0, cv.BORDER_CONSTANT, value=cval)
    center = center + hpad - radius
    new_height = 2 * radius
    dewarped = [padded[c : c + new_height, i] for i, c in enumerate(center)]
    return np.swapaxes(np.array(dewarped, dtype=np.uint8), 1, 0)


def _measure_centerline(
    line: np.ndarray,
    *,
    range_: int,
    smoothness: float,
    extra: float,
) -> tuple[np.ndarray, int]:
    height, width = line.shape
    smoothed = cv.GaussianBlur(
        line,
        (0, 0),
        sigmaX=height * smoothness,
        sigmaY=height * 0.5,
        borderType=cv.BORDER_CONSTANT,
    )
    smoothed += 0.001 * cv.blur(
        smoothed,
        (width, int(height * 0.5)),
        borderType=cv.BORDER_CONSTANT,
    )

    argmax = np.argmax(smoothed, axis=0).astype(np.uint16)
    kernel = cv.getGaussianKernel(int((8.0 * height * extra) + 1), height * extra)
    center = cv.filter2D(argmax, cv.CV_16U, kernel, borderType=cv.BORDER_REFLECT).flatten()

    deltas = abs(np.arange(height)[:, np.newaxis] - center[np.newaxis, :])
    mad = np.mean(deltas[line != 0])
    radius = int(1 + range_ * mad)
    return center, radius
