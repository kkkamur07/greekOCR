"""Focused tests for the BLLA preprocessing, decoder and runner path.

The graph itself, and the Kraken-oracle comparisons that used it, moved to
``tests/export`` with the Torch modules under ADR 0006. What is left here is
what a researcher's install actually contains: NumPy preprocessing, the
Torch-free decoder, and the runner that ties them to the ONNX session.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from inference.architectures.blla.blla_decoder import decode_blla_heatmaps
from inference.architectures.blla.blla_decoder.common import resize_heatmaps_nearest
from inference.architectures.blla.blla_preprocessing import (
    MAX_WIDTH_TO_HEIGHT_RATIO,
    preprocess_blla_image,
)


def test_blla_preprocessing_matches_fixed_height_rgb_inversion() -> None:
    image = Image.new("RGB", (20, 10), (0, 64, 255))

    prepared = preprocess_blla_image(image)

    assert prepared.array.shape == (3, 1800, 3600)
    assert prepared.array.dtype == np.float32
    np.testing.assert_allclose(
        prepared.array[:, 0, 0], np.array([1.0, 0.7490196, 0.0], dtype=np.float32), atol=1e-6
    )
    assert prepared.scaled_gray.shape == (1800, 3600)
    assert prepared.scale_xy == pytest.approx((20 / 3600, 10 / 1800))


def test_blla_preprocessing_caps_extreme_aspect_ratio_width() -> None:
    """A panorama within the pixel admission cap must not produce an unbounded array."""
    input_height = 90
    capped_width = input_height * MAX_WIDTH_TO_HEIGHT_RATIO
    image = Image.new("RGB", (4000, 10), (255, 255, 255))

    prepared = preprocess_blla_image(image, input_height=input_height)

    assert prepared.array.shape == (3, input_height, capped_width)
    # Coordinates still map back to source space through scale_xy.
    assert prepared.scale_xy == pytest.approx((4000 / capped_width, 10 / input_height))


def test_blla_preprocessing_is_proportional_below_the_width_cap() -> None:
    image = Image.new("RGB", (20, 10), (255, 255, 255))

    prepared = preprocess_blla_image(image, input_height=90)

    assert prepared.array.shape == (3, 90, 180)


def test_blla_decoder_turns_separator_ridge_into_line_polygon() -> None:
    heatmaps = np.zeros((4, 80, 100), dtype=np.float32)
    heatmaps[0, 38:43, 10:91] = 1.0
    heatmaps[1, 38:43, 10:91] = 1.0
    heatmaps[2, 38:43, 10:91] = 1.0
    heatmaps[3, 30:51, 10:91] = 1.0

    lines = decode_blla_heatmaps(heatmaps, image_size=(100, 80))

    assert len(lines) == 1
    assert lines[0].baseline[0][0] == pytest.approx(10.0)
    assert lines[0].baseline[-1][0] == pytest.approx(90.0)
    ys = [point[1] for point in lines[0].polygon]
    assert min(ys) == pytest.approx(30.0)
    assert max(ys) == pytest.approx(50.0)


def test_nearest_resize_repeats_source_pixels_rather_than_blending() -> None:
    """The decoder's upsample is index arithmetic, so no new values may appear.

    This replaced ``torch.nn.functional.interpolate`` when Torch left the
    runtime. Interpolation with any other mode would invent intermediate
    probabilities and move the 0.17 threshold's boundary.
    """
    heatmaps = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2)

    resized = resize_heatmaps_nearest(heatmaps, height=4, width=6)

    assert resized.shape == (4, 4, 6)
    assert set(np.unique(resized)) <= set(np.unique(heatmaps))
    # Row 0 of channel 0 is its two source pixels, each repeated three times.
    np.testing.assert_array_equal(resized[0, 0], np.array([0, 0, 0, 1, 1, 1], dtype=np.float32))


# `test_run_model_returns_a_blla_response_for_a_real_image` stood here and ran the
# published artifact through `run_model`, asserting `blocks == 1` and `len(lines) > 10`.
# `test_onnx_runtime.py::test_segment_runs_the_graph_on_real_weights` makes the same two
# assertions on the same artifact and the same page; the `run_model` dispatch increment is
# covered by `test_transcribe_batch_isolation.py`'s runner fixture and end to end by
# `test_cli_run.py`. Cutting it removes a second full model execution from the unit lane.
