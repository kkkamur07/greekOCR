"""Unit tests for line baseline helpers."""

from backend.annotation.application.line_geometry import (
    baseline_matches_polygon,
    default_baseline_from_polygon,
    geometry_points,
    resolve_line_baseline_and_mask,
)


def test_default_baseline_uses_bottom_edge() -> None:
    polygon = [[0, 0], [10, 0], [10, 8], [0, 8]]
    assert default_baseline_from_polygon(polygon) == [[0, 8], [10, 8]]


def test_resolve_preserves_kraken_baseline_on_points_update() -> None:
    polygon = [[0, 0], [10, 0], [10, 8], [0, 8]]
    kraken_baseline = {"points": [[1, 7], [5, 7.5], [9, 7]]}
    kraken_mask = {"points": polygon}

    baseline, mask = resolve_line_baseline_and_mask(
        points=[[1, 1], [11, 1], [11, 9], [1, 9]],
        payload_baseline=None,
        payload_mask=None,
        existing_baseline=kraken_baseline,
        existing_mask=kraken_mask,
    )

    assert baseline == kraken_baseline
    assert mask == kraken_mask


def test_resolve_replaces_polygon_copy_baseline() -> None:
    polygon = [[0, 0], [10, 0], [10, 8], [0, 8]]
    copied = {"points": polygon}

    baseline, mask = resolve_line_baseline_and_mask(
        points=polygon,
        payload_baseline=None,
        payload_mask=None,
        existing_baseline=copied,
        existing_mask=None,
    )

    assert geometry_points(baseline) == [[0, 8], [10, 8]]
    assert baseline_matches_polygon(geometry_points(copied), polygon)
