"""Inference-owned BLLA heatmap, baseline, and polygon decoding."""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage import filters

from inference.architectures.blla.blla_decoder.common import as_heatmaps
from shapely import geometry as geom

from inference.architectures.isolation import reraise_if_none_survived
from inference.architectures.blla.blla_decoder.lines import (
    is_in_region,
    reading_order_indices,
    vectorize_lines,
    vectorize_regions,
)
from inference.architectures.blla.blla_decoder.polygon import calculate_polygonal_environment
from inference.architectures.blla.blla_decoder.simple import decode_simple_heatmaps
from inference.architectures.blla.blla_decoder.types import DecodedBLLALine

logger = logging.getLogger(__name__)

__all__ = ["DecodedBLLALine", "decode_blla_heatmaps"]


def decode_blla_heatmaps(
    heatmaps: np.ndarray,
    *,
    image_size: tuple[int, int],
    threshold: float = 0.17,
    min_length: float = 5.0,
    raw_logits: bool = False,
    scaled_gray: np.ndarray | None = None,
) -> list[DecodedBLLALine]:
    """Decode BLLA channels into image-space baselines and polygons.

    Production inference follows the reference BLLA decoder when
    ``scaled_gray`` is supplied. The small connected-component path remains
    available for focused decoder tests that provide synthetic probabilities
    without an image.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    if scaled_gray is None:
        return decode_simple_heatmaps(
            heatmaps,
            image_size=image_size,
            threshold=threshold,
            min_length=min_length,
        )
    return _decode_reference_pipeline(
        heatmaps,
        image_size=image_size,
        threshold=threshold,
        raw_logits=raw_logits,
        scaled_gray=scaled_gray,
        min_length=min_length,
    )


def _decode_reference_pipeline(
    heatmaps: np.ndarray,
    *,
    image_size: tuple[int, int],
    threshold: float,
    raw_logits: bool,
    scaled_gray: np.ndarray,
    min_length: float,
) -> list[DecodedBLLALine]:
    """Run the reference heatmap, skeleton, and polygonization sequence."""

    values = as_heatmaps(heatmaps)
    scaled_height, scaled_width = scaled_gray.shape
    # ``interpolate`` without a mode is nearest-neighbour for a 4D tensor, which
    # is what the reference BLLA decoder uses.
    with torch.inference_mode():
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(values).unsqueeze(0),
            size=(scaled_height, scaled_width),
        )[0]
        probabilities = (torch.sigmoid(resized) if raw_logits else resized).numpy()
    baselines = vectorize_lines(
        probabilities[:3],
        threshold=threshold,
        min_length=min_length,
    )
    regions_scaled = vectorize_regions(probabilities[3])
    image_features = gaussian_filter(filters.sobel(scaled_gray), 0.5)
    bounds = np.asarray((scaled_width, scaled_height), dtype=float) - 1
    scale_xy = np.asarray((image_size[0] / scaled_width, image_size[1] / scaled_height))
    regions_original = [
        (np.asarray(region) * scale_xy).astype("uint").tolist() for region in regions_scaled
    ]
    # Kraken round-trips regions through original-image coordinates before
    # using them as polygonization supplements.
    regions_for_polygonization = [
        (np.asarray(region) * (1 / scale_xy)).astype("uint").tolist() for region in regions_original
    ]

    decoded: list[DecodedBLLALine] = []
    # Polygonization is per-line geometry work and it raises by design: a
    # baseline whose environment closes into an invalid polygon, or whose ray
    # never meets a boundary, is a ``ValueError`` from ``polygon``. Under
    # ``architectures.isolation`` that must cost its own line and not the other
    # thirty-nine on the page, so the first failure is held for the all-failed
    # verdict rather than propagated on the spot.
    first_failure: Exception | None = None
    for index, baseline in enumerate(baselines):
        try:
            supplementary_objects = baselines[:index] + baselines[index + 1 :]
            baseline_line = geom.LineString(baseline)
            supplementary_objects.extend(
                region
                for region in regions_for_polygonization
                if is_in_region(baseline_line, geom.Polygon(region))
            )
            polygon = calculate_polygonal_environment(
                baseline=baseline,
                supplementary_objects=supplementary_objects,
                image_features=image_features,
                bounds=bounds,
                topline=False,
            )
        except Exception as error:  # noqa: BLE001 - one bad baseline is not a bad page
            first_failure = first_failure or error
            logger.warning(
                "BLLA polygonization failed (baseline_index=%s, baseline_points=%s)",
                index,
                len(baseline),
                exc_info=error,
            )
            continue
        scaled_baseline = (np.asarray(baseline) * scale_xy).astype("int").tolist()
        scaled_polygon = (np.asarray(polygon) * scale_xy).astype("int").tolist()
        decoded.append(
            DecodedBLLALine(
                baseline=scaled_baseline,
                polygon=scaled_polygon,
            )
        )

    # A page that polygonized nothing *and* failed at least once is a failed
    # run, not a blank page; the rule and the reason live in
    # ``architectures.isolation``.
    reraise_if_none_survived(survivors=len(decoded), first_failure=first_failure)

    order = reading_order_indices(
        [line.baseline for line in decoded],
        regions_original,
    )
    return [decoded[index] for index in order]
