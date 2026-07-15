"""Tests for Kraken ceiling refinement with Otsu and polygon simplification."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from inference.architectures import kraken
from inference.preprocessing.segment_geometry import MIN_VERTEX_SPACING_PX, distance
from inference.preprocessing.segment_refinement import refine_segment, refine_segment_candidates
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

    result = refine_segment(image, dense_ceiling)
    x0, y0, x1, y1 = _bbox(result.points)

    assert len(result.points) <= 80
    assert len(result.points) < len(dense_ceiling)
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

    result = refine_segment(image, ceiling)
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


# --- No-ink fallback ---
# Tests dense ceiling simplifies when Otsu finds no contour. Does not test real manuscript pages.


def test_refine_segment_falls_back_to_clean_ceiling_without_ink() -> None:
    image = Image.new("RGB", (120, 80), "white")
    dense_ceiling = _dense_rectangle(10, 10, 110, 70)

    result = refine_segment(image, dense_ceiling)

    assert len(result.points) < len(dense_ceiling)
    assert result.metadata["simplification_status"] == "no_otsu_contour"


# --- Kraken adapter integration (stubbed BLLA) ---
# Tests adapter stores ceiling, refined points, and metadata. Does not run real Kraken inference.


def test_kraken_adapter_preserves_raw_ceiling_and_stores_refined_points(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Post-BLLA refinement only.

    Stubs ``_run_blla_segment`` to inject a fixed dense ceiling so this unit
    test can assert Otsu/simplify metadata without loading Kraken weights.
    Live BLLA is covered by ``@pytest.mark.ml`` run/worker/e2e suites.
    """
    image, _ = _synthetic_ink_fixture()
    dense_ceiling = _dense_rectangle(20, 25, 200, 90)
    model_path = tmp_path / "model.mlmodel"
    model_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(kraken, "_load_segmentation_model", lambda _: object())
    monkeypatch.setattr(
        kraken,
        "_run_blla_segment",
        lambda _image, _model: {
            "lines": [
                {
                    "baseline": [[55.0, 57.0], [165.0, 57.0]],
                    "boundary": dense_ceiling,
                }
            ]
        },
    )

    response = kraken.run_kraken_segment(
        _image_bytes(image),
        model_path=model_path,
        params={"use_otsu_refinement": True, "otsu_sphere_radius": 6},
    )

    line = response.lines[0]
    assert line.kraken_ceiling == dense_ceiling
    assert line.points != dense_ceiling
    assert len(line.points) <= 80
    assert line.mask == {"points": line.points}
    assert line.source_metadata["adapter"] == "kraken"
    assert line.source_metadata["raw_point_count"] == len(dense_ceiling)
    assert line.source_metadata["simplified_point_count"] == len(line.points)
    assert line.source_metadata["otsu_margin_px"] == 6


# --- Kraken adapter without Otsu ---
# Tests boundary simplification when Otsu refinement is disabled.


def test_kraken_adapter_uses_raw_boundaries_when_otsu_refinement_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Post-BLLA simplify only — BLLA stub justified as in the Otsu test above."""
    image, _ = _synthetic_ink_fixture()
    dense_ceiling = _dense_rectangle(20, 25, 200, 90)
    model_path = tmp_path / "model.mlmodel"
    model_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(kraken, "_load_segmentation_model", lambda _: object())
    monkeypatch.setattr(
        kraken,
        "_run_blla_segment",
        lambda _image, _model: {
            "lines": [
                {
                    "baseline": [[55.0, 57.0], [165.0, 57.0]],
                    "boundary": dense_ceiling,
                }
            ]
        },
    )

    response = kraken.run_kraken_segment(_image_bytes(image), model_path=model_path)

    line = response.lines[0]
    assert line.kraken_ceiling == dense_ceiling
    assert line.points != dense_ceiling
    assert len(line.points) < len(dense_ceiling)
    _assert_adjacent_points_are_spaced(line.points)
    assert line.source_metadata["simplification_status"] == "kraken_boundary_simplified"
    assert line.source_metadata["raw_point_count"] == len(dense_ceiling)
    assert line.source_metadata["simplified_point_count"] == len(line.points)


# --- Kraken adapter line splitting ---
# Tests oversized refined lines split into multiple SegmentLine rows.


def test_kraken_adapter_splits_oversized_refined_line(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Post-BLLA split only — BLLA stub justified as in the Otsu test above."""
    image, ceiling = _merged_two_line_fixture()
    model_path = tmp_path / "model.mlmodel"
    model_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(kraken, "_load_segmentation_model", lambda _: object())
    monkeypatch.setattr(
        kraken,
        "_run_blla_segment",
        lambda _image, _model: {
            "lines": [
                {
                    "baseline": [[35.0, 77.0], [190.0, 77.0]],
                    "boundary": ceiling,
                }
            ]
        },
    )

    response = kraken.run_kraken_segment(
        _image_bytes(image),
        model_path=model_path,
        params={"use_otsu_refinement": True, "split_large_lines": True},
    )

    assert len(response.lines) == 2
    assert [line.order for line in response.lines] == [0, 1]
    assert all(line.kraken_ceiling == ceiling for line in response.lines)
    assert all(line.source_metadata["split_count"] == 2 for line in response.lines)
    assert response.lines[0].points != response.lines[1].points
