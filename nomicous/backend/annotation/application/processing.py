"""Platform-owned Processing pipeline for exported line images."""

from collections.abc import Callable

import numpy as np

SUPPORTED_STEPS = {"rectify"}
StepCallback = Callable[[str], None]


def _mask_bbox_rectify(page_image: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = page_image.shape[:2]
    xs, ys = points[:, 0], points[:, 1]
    x0 = int(np.floor(np.clip(xs.min(), 0, w - 1)))
    y0 = int(np.floor(np.clip(ys.min(), 0, h - 1)))
    x1 = int(np.ceil(np.clip(xs.max(), x0 + 1, w)))
    y1 = int(np.ceil(np.clip(ys.max(), y0 + 1, h)))

    crop = page_image[y0:y1, x0:x1].copy()
    crop_h, crop_w = crop.shape[:2]

    try:
        import cv2

        shifted = points.copy()
        shifted[:, 0] -= x0
        shifted[:, 1] -= y0
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)
        white = np.full_like(crop, 255)
        return np.where(mask[:, :, None] > 0, crop, white)
    except ImportError:
        return crop


def rectify(page_image: np.ndarray, segment: dict) -> np.ndarray:
    raw_points = segment.get("points") or []
    points = np.array(raw_points, dtype=np.float32)
    if points.size == 0:
        raise ValueError("Segment has no points")
    if len(points) < 3:
        xs, ys = points[:, 0], points[:, 1]
        x0, y0 = int(max(xs.min(), 0)), int(max(ys.min(), 0))
        x1 = int(min(xs.max(), page_image.shape[1]))
        y1 = int(min(ys.max(), page_image.shape[0]))
        return page_image[y0:y1, x0:x1]

    return _mask_bbox_rectify(page_image, points)


def apply_step(image: np.ndarray, segment: dict, step: str) -> np.ndarray:
    if step not in SUPPORTED_STEPS:
        raise ValueError(f"Unsupported processing step: {step}")
    if step == "rectify":
        return rectify(image, segment)
    raise ValueError(f"Unsupported processing step: {step}")


def process(
    image: np.ndarray,
    segment: dict,
    steps: list[str],
    *,
    on_step: StepCallback | None = None,
) -> np.ndarray:
    result = image
    for step in steps:
        if on_step is not None:
            on_step(step)
        result = apply_step(result, segment, step)
    return result
