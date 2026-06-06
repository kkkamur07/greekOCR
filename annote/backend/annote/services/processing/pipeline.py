"""Pluggable processing pipeline."""

from collections.abc import Callable

import numpy as np

from annote.services.processing.binarize import binarize
from annote.services.processing.rectify import rectify

SUPPORTED_STEPS = {"rectify", "binarize"}
StepCallback = Callable[[str], None]


def apply_step(image: np.ndarray, segment: dict, step: str) -> np.ndarray:
    """Apply one supported processing step."""
    if step not in SUPPORTED_STEPS:
        raise ValueError(f"Unsupported processing step: {step}")
    if step == "rectify":
        return rectify(image, segment)
    if step == "binarize":
        return binarize(image)
    raise ValueError(f"Unsupported processing step: {step}")


def process(
    image: np.ndarray,
    segment: dict,
    steps: list[str],
    *,
    on_step: StepCallback | None = None,
) -> np.ndarray:
    """Run ordered processing steps on a segment crop."""
    result = image
    for step in steps:
        if on_step is not None:
            on_step(step)
        result = apply_step(result, segment, step)
    return result
