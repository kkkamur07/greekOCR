"""Polygon preprocessing."""

from annote.services.processing.polygon import merge_close_polygon_points


def test_merge_close_polygon_points_drops_dense_runs():
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [20.0, 0.0],
        [20.0, 10.0],
        [0.0, 10.0],
    ]
    merged = merge_close_polygon_points(points, min_distance=5.0)
    assert merged == [[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]]


def test_merge_close_polygon_points_drops_duplicate_closing_vertex():
    points = [
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 5.0],
        [0.0, 5.0],
        [0.1, 0.1],
    ]
    merged = merge_close_polygon_points(points, min_distance=2.0)
    assert merged == [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]


def test_merge_close_polygon_points_keeps_original_when_too_few_remain():
    points = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]
    merged = merge_close_polygon_points(points, min_distance=5.0)
    assert merged == points


def test_merge_close_polygon_points_merges_close_consecutive_vertices():
    points = [
        [1078.0, 83.0],
        [1072.0, 83.0],
        [1066.0, 83.0],
        [1061.0, 92.0],
        [1061.0, 98.0],
        [1080.0, 97.0],
        [1080.0, 91.0],
    ]
    merged = merge_close_polygon_points(points)
    assert [1072.0, 83.0] not in merged
    assert len(merged) < len(points)


def test_default_duplicate_spacing_matches_manuscript_calibration():
    from annote.services.processing.polygon import DUPLICATE_VERTEX_SPACING

    assert DUPLICATE_VERTEX_SPACING == 6.0


def test_merge_close_polygon_points_keeps_short_rectangle_corners():
    points = [
        [0.0, 0.0],
        [50.0, 0.0],
        [50.0, 8.0],
        [0.0, 8.0],
    ]
    merged = merge_close_polygon_points(points, min_distance=10.0)
    assert merged == points


def test_kraken_boundary_preprocessing():
    from annote.services.kraken_segment import kraken_lines_to_segments
    from types import SimpleNamespace

    boundary = [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [50, 0],
        [50, 8],
        [0, 8],
    ]
    segments = kraken_lines_to_segments([SimpleNamespace(boundary=boundary)])
    assert len(segments) == 1
    assert segments[0].points == [[0.0, 0.0], [50.0, 0.0], [50.0, 8.0], [0.0, 8.0]]
