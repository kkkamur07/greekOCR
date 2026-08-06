"""Tests for Kraken ceiling refinement with Otsu and polygon simplification."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from inference.architectures.blla import blla
from inference.architectures.blla.blla_decoder import DecodedBLLALine
from inference.preprocessing.segment_geometry import (
    MIN_VERTEX_SPACING_PX,
    bottom_edge_baseline,
    clip_baseline_to_x_span,
    distance,
)
from inference.preprocessing.segment_refinement import refine_segment_candidates
from PIL import Image


def _synthetic_ink_fixture() -> tuple[Image.Image, list[list[float]]]:
    image = Image.new("RGB", (220, 120), "white")
    pixels = image.load()
    for y in range(48, 66):
        for x in range(55, 166):
            pixels[x, y] = (0, 0, 0)
    ceiling = [[20.0, 25.0], [200.0, 25.0], [200.0, 90.0], [20.0, 90.0]]
    return image, ceiling


def _gapped_character_line_fixture() -> tuple[Image.Image, list[list[float]]]:
    image = Image.new("RGB", (420, 120), "white")
    pixels = image.load()
    for char_x in (40, 80, 125, 175, 230, 285, 340):
        for y in range(45, 72):
            for x in range(char_x, char_x + 18):
                pixels[x, y] = (0, 0, 0)
    ceiling = [[20.0, 30.0], [390.0, 30.0], [390.0, 90.0], [20.0, 90.0]]
    return image, ceiling


def _merged_two_line_fixture() -> tuple[Image.Image, list[list[float]]]:
    image = Image.new("RGB", (240, 160), "white")
    pixels = image.load()
    for y in range(40, 58):
        for x in range(45, 190):
            pixels[x, y] = (0, 0, 0)
    for y in range(96, 114):
        for x in range(35, 180):
            pixels[x, y] = (0, 0, 0)
    ceiling = [[20.0, 20.0], [210.0, 20.0], [210.0, 135.0], [20.0, 135.0]]
    return image, ceiling


def _dense_rectangle(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    step: int = 1,
) -> list[list[float]]:
    points: list[list[float]] = []
    for x in range(x0, x1 + 1, step):
        points.append([float(x), float(y0)])
    for y in range(y0 + step, y1 + 1, step):
        points.append([float(x1), float(y)])
    for x in range(x1 - step, x0 - 1, -step):
        points.append([float(x), float(y1)])
    for y in range(y1 - step, y0, -step):
        points.append([float(x0), float(y)])
    return points


def _refine_one(image: Image.Image, ceiling: list[list[float]]):
    """Refine a ceiling without splitting it, the way the runner does when asked not to."""
    return refine_segment_candidates(image, ceiling, split_large_lines=False)[0]


def _bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _assert_adjacent_points_are_spaced(points: list[list[float]]) -> None:
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        assert distance(point, next_point) >= MIN_VERTEX_SPACING_PX


def _assert_points_inside_ceiling(points: list[list[float]], ceiling: list[list[float]]) -> None:
    import cv2

    polygon = np.array(ceiling, dtype=np.float32)
    for x, y in points:
        assert cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def _image_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


# --- Otsu refinement ---
# Tests contour refinement and simplification inside a ceiling. Does not load Kraken weights.


def test_refine_segment_runs_otsu_inside_ceiling_and_simplifies_dense_contour() -> None:
    image, ceiling = _synthetic_ink_fixture()
    dense_ceiling = _dense_rectangle(20, 25, 200, 90)

    result = _refine_one(image, dense_ceiling)
    x0, y0, x1, y1 = _bbox(result.points)

    assert len(result.points) <= 80
    assert len(result.points) < len(dense_ceiling)
    # The point of simplification, and the only place it is checked. The helper
    # was defined and never called, so the minimum-vertex-spacing invariant read
    # as covered while nothing evaluated it - a one-pixel-apart contour would
    # have satisfied every count assertion above.
    _assert_adjacent_points_are_spaced(result.points)
    assert result.metadata["raw_point_count"] == len(dense_ceiling)
    assert result.metadata["simplified_point_count"] == len(result.points)
    assert result.metadata["simplification_status"] in {
        "simplified",
        "quality_gate_stopped",
        "max_points_not_reached",
    }
    assert y0 <= 48.0
    assert y1 >= 65.0
    assert x0 >= 20.0
    assert x1 <= 200.0
    _assert_points_inside_ceiling(result.points, ceiling)


# --- Gapped ink components ---
# Tests refinement spans separated character blobs. Does not split merged lines.


def test_refine_segment_encloses_gapped_ink_components() -> None:
    image, ceiling = _gapped_character_line_fixture()

    result = _refine_one(image, ceiling)
    x0, _, x1, _ = _bbox(result.points)

    assert x0 <= 40.0
    assert x1 >= 358.0
    _assert_points_inside_ceiling(result.points, ceiling)


# --- Multi-line split ---
# Tests merged bands split into separate line candidates. Does not call the Kraken model.


def test_refine_segment_candidates_splits_merged_vertical_bands() -> None:
    image, ceiling = _merged_two_line_fixture()

    results = refine_segment_candidates(
        image,
        ceiling,
        split_large_lines=True,
        split_vertical_gap_px=12,
    )

    assert len(results) == 2
    assert [result.metadata["split_index"] for result in results] == [0, 1]
    assert all(result.metadata["split_count"] == 2 for result in results)
    assert all(result.baseline is not None for result in results)
    first_bbox = _bbox(results[0].points)
    second_bbox = _bbox(results[1].points)
    assert first_bbox[3] < second_bbox[1]
    assert first_bbox[1] <= 40.0
    assert second_bbox[3] >= 113.0


def test_split_band_holding_the_decoded_baseline_keeps_it() -> None:
    """The band the decoder measured must not be handed a substitute."""
    image, ceiling = _merged_two_line_fixture()
    baseline = [[40.0, 112.0], [180.0, 112.0]]

    results = refine_segment_candidates(
        image,
        ceiling,
        baseline=baseline,
        split_large_lines=True,
        split_vertical_gap_px=12,
    )

    lower = results[1]
    assert lower.metadata["baseline_source"] == "decoder"
    assert lower.baseline == baseline


def test_split_band_without_a_decoded_baseline_gets_one_off_its_bottom_edge() -> None:
    """A mid-height segment is a strike-through, not a text baseline."""
    image, ceiling = _merged_two_line_fixture()

    results = refine_segment_candidates(
        image,
        ceiling,
        baseline=[[40.0, 112.0], [180.0, 112.0]],
        split_large_lines=True,
        split_vertical_gap_px=12,
    )

    upper = results[0]
    _, y0, _, y1 = _bbox(upper.points)
    assert upper.metadata["baseline_source"] == "bottom_edge"
    assert upper.baseline is not None
    assert all(point[1] == pytest.approx(y1) for point in upper.baseline)
    assert y1 > (y0 + y1) / 2.0


def test_decoded_baseline_is_clipped_to_the_band_that_holds_it() -> None:
    image, ceiling = _merged_two_line_fixture()
    # Runs the full page width; only the middle of it is over the lower band.
    baseline = [[0.0, 112.0], [239.0, 112.0]]

    results = refine_segment_candidates(
        image,
        ceiling,
        baseline=baseline,
        split_large_lines=True,
        split_vertical_gap_px=12,
    )

    lower = results[1]
    x0, _, x1, _ = _bbox(lower.points)
    assert lower.metadata["baseline_source"] == "decoder"
    assert lower.baseline is not None
    assert min(point[0] for point in lower.baseline) >= x0
    assert max(point[0] for point in lower.baseline) <= x1


def test_clipping_a_baseline_follows_its_slope_instead_of_flattening_at_the_cut() -> None:
    clipped = clip_baseline_to_x_span([[0.0, 0.0], [100.0, 100.0]], 20.0, 60.0)

    assert clipped == [[20.0, 20.0], [60.0, 60.0]]


def test_a_baseline_that_misses_the_span_clips_to_nothing() -> None:
    assert clip_baseline_to_x_span([[0.0, 5.0], [10.0, 5.0]], 40.0, 80.0) == []


def test_bottom_edge_baseline_sits_on_the_bottom_not_the_middle() -> None:
    assert bottom_edge_baseline([[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]]) == [
        [0.0, 8.0],
        [10.0, 8.0],
    ]


def test_unsplit_line_passes_the_decoded_baseline_through_untouched() -> None:
    image, ceiling = _synthetic_ink_fixture()
    baseline = [[55.0, 65.0], [165.0, 65.0]]

    results = refine_segment_candidates(image, ceiling, baseline=baseline, split_large_lines=False)

    assert len(results) == 1
    assert results[0].baseline == baseline
    assert results[0].metadata["baseline_source"] == "decoder"


# --- No-ink fallback ---
# Tests dense ceiling simplifies when Otsu finds no contour. Does not test real manuscript pages.


def test_refine_segment_falls_back_to_clean_ceiling_without_ink() -> None:
    image = Image.new("RGB", (120, 80), "white")
    dense_ceiling = _dense_rectangle(10, 10, 110, 70)

    result = _refine_one(image, dense_ceiling)

    assert len(result.points) < len(dense_ceiling)
    assert result.metadata["simplification_status"] == "no_otsu_contour"


# --- BLLA adapter integration (stubbed session) ---


class _FakeBLLASession:
    """A session-shaped stand-in whose output is *not* what these tests assert on.

    These cases exercise refinement geometry, so the decoded lines are injected
    directly and the graph only has to produce a correctly shaped array. The
    shape is derived from the input rather than fixed, because the adapter
    validates the logits it gets back (ADR 0006) and a constant would stop
    exercising that check.
    """

    def run(self, _output_names, feeds):
        values = next(iter(feeds.values()))
        width = max(1, values.shape[-1] // 4)
        return [np.zeros((1, 4, 450, width), dtype=np.float32)]


def _stub_blla(monkeypatch, decoded: DecodedBLLALine) -> None:
    monkeypatch.setattr(
        blla, "_load_blla_session", lambda *_args, **_kwargs: (_FakeBLLASession(), "input")
    )
    monkeypatch.setattr(
        "inference.architectures.blla.blla_runtime.decode_blla_heatmaps",
        lambda *_args, **_kwargs: [decoded],
    )


def test_blla_adapter_preserves_legacy_ceiling_and_neutral_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image, _ = _synthetic_ink_fixture()
    dense_ceiling = _dense_rectangle(20, 25, 200, 90)
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"stub")
    _stub_blla(
        monkeypatch,
        DecodedBLLALine(
            baseline=[[55.0, 57.0], [165.0, 57.0]],
            polygon=dense_ceiling,
        ),
    )

    response = blla.run_blla_segment(
        _image_bytes(image),
        model_path=model_path,
        params={"use_otsu_refinement": True, "otsu_sphere_radius": 6},
    )

    line = response.lines[0]
    assert line.kraken_ceiling == dense_ceiling
    assert line.points != dense_ceiling
    assert line.source_metadata["adapter"] == "blla"
    assert line.source_metadata["raw_point_count"] == len(dense_ceiling)
    assert line.source_metadata["otsu_margin_px"] == 6


# `test_blla_adapter_splits_oversized_refined_line` stood here. The split itself is
# `test_refine_segment_candidates_splits_merged_vertical_bands` above, on the same
# `_merged_two_line_fixture`; this re-asserted it through the adapter behind a
# `_FakeBLLASession`, adding only `line.order == [0, 1]` and the `kraken_ceiling`
# passthrough that `test_blla_adapter_preserves_legacy_ceiling_and_neutral_metadata`
# already covers.


# --- Per-line refinement isolation ---
# Tests one failing line does not discard the page. Does not exercise real weights.


def _stub_blla_lines(monkeypatch, decoded: list[DecodedBLLALine]) -> None:
    monkeypatch.setattr(
        blla, "_load_blla_session", lambda *_args, **_kwargs: (_FakeBLLASession(), "input")
    )
    monkeypatch.setattr(
        "inference.architectures.blla.blla_runtime.decode_blla_heatmaps",
        lambda *_args, **_kwargs: decoded,
    )


def test_blla_adapter_keeps_the_page_when_one_line_refinement_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image, ceiling = _synthetic_ink_fixture()
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"stub")
    doomed = [[30.0, 95.0], [190.0, 95.0], [190.0, 110.0], [30.0, 110.0]]
    _stub_blla_lines(
        monkeypatch,
        [
            DecodedBLLALine(baseline=[[55.0, 57.0], [165.0, 57.0]], polygon=ceiling),
            DecodedBLLALine(baseline=[[35.0, 102.0], [185.0, 102.0]], polygon=doomed),
        ],
    )

    def refine(_image, contour, **_kwargs):
        if contour == doomed:
            raise RuntimeError("OpenCV rejected the contour")
        return refine_segment_candidates(_image, contour, **_kwargs)

    monkeypatch.setattr(
        "inference.architectures.blla.blla_runtime.refine_segment_candidates", refine
    )

    response = blla.run_blla_segment(
        _image_bytes(image),
        model_path=model_path,
        params={"use_otsu_refinement": True},
    )

    assert len(response.lines) == 1
    assert response.lines[0].source_metadata["raw_order"] == 0
    assert len(response.blocks) == 1


def test_blla_adapter_fails_when_every_line_refinement_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """An empty page is a worse answer than an error when nothing could refine."""
    image, ceiling = _synthetic_ink_fixture()
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"stub")
    _stub_blla_lines(
        monkeypatch,
        [DecodedBLLALine(baseline=[[55.0, 57.0], [165.0, 57.0]], polygon=ceiling)],
    )
    monkeypatch.setattr(
        "inference.architectures.blla.blla_runtime.refine_segment_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("OpenCV rejected")),
    )

    with pytest.raises(RuntimeError, match="OpenCV rejected"):
        blla.run_blla_segment(
            _image_bytes(image),
            model_path=model_path,
            params={"use_otsu_refinement": True},
        )
