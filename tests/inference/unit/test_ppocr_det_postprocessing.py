"""PP-OCRv6 DB postprocess over synthetic probability maps."""

from __future__ import annotations

import cv2
import numpy as np

from nomikos_inference.architectures.ppocr_det.postprocessing import (
    detect_lines,
    order_quad_clockwise,
    unclip,
)


def _rotated_rect_map(
    width: int,
    height: int,
    rects: list[tuple[tuple[float, float], tuple[float, float], float]],
) -> np.ndarray:
    prob = np.zeros((height, width), dtype=np.float32)
    for center, size, angle in rects:
        corners = cv2.boxPoints((center, size, angle)).astype(np.int32)
        cv2.fillPoly(prob, [corners], 1.0)
    return prob


def _expected_expanded_corners(
    center: tuple[float, float],
    size: tuple[float, float],
    angle: float,
    unclip_ratio: float = 1.4,
) -> np.ndarray:
    width, height = size
    distance = width * height * unclip_ratio / (2 * (width + height))
    half = np.array([width / 2 + distance, height / 2 + distance])
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners = np.array(
        [[half[0], half[1]], [-half[0], half[1]], [-half[0], -half[1]], [half[0], -half[1]]]
    )
    return corners @ rotation.T + np.array(center)


def _corner_error(found: list[list[float]], expected: np.ndarray) -> float:
    found_corners = np.asarray(found, dtype=np.float64)
    best = float("inf")
    for shift in range(4):
        rolled = np.roll(found_corners, shift, axis=0)
        best = min(best, float(np.abs(rolled - expected).max()))
    return best


def test_three_rotated_rectangles_return_expanded_quads() -> None:
    rects = [
        ((180.0, 150.0), (220.0, 36.0), 12.0),
        ((450.0, 320.0), (200.0, 32.0), -8.0),
        ((250.0, 500.0), (240.0, 40.0), 5.0),
    ]
    prob = _rotated_rect_map(640, 640, rects)

    quads = detect_lines(prob, orig_width=640, orig_height=640, ratio_h=1.0, ratio_w=1.0)

    assert len(quads) == 3
    matched = sorted(quads, key=lambda quad: np.mean([point[1] for point in quad.points]))
    for quad, (center, size, angle) in zip(
        matched, sorted(rects, key=lambda rect: rect[0][1]), strict=True
    ):
        expected = _expected_expanded_corners(center, size, angle)
        assert _corner_error(quad.points, expected) <= 2.0
        assert quad.score > 0.9


def test_blob_below_box_thresh_is_dropped() -> None:
    prob = np.zeros((128, 128), dtype=np.float32)
    prob[40:80, 30:90] = 0.3

    quads = detect_lines(prob, orig_width=128, orig_height=128, ratio_h=1.0, ratio_w=1.0)

    assert quads == []


def test_two_pixel_speck_is_dropped() -> None:
    prob = np.zeros((128, 128), dtype=np.float32)
    prob[60:62, 60:62] = 1.0

    quads = detect_lines(prob, orig_width=128, orig_height=128, ratio_h=1.0, ratio_w=1.0)

    assert quads == []


def test_coordinates_map_back_with_per_axis_ratios() -> None:
    prob = np.zeros((160, 160), dtype=np.float32)
    prob[40:80, 30:90] = 1.0

    quads = detect_lines(prob, orig_width=640, orig_height=320, ratio_h=0.5, ratio_w=0.25)

    assert len(quads) == 1
    xs = [point[0] for point in quads[0].points]
    ys = [point[1] for point in quads[0].points]
    # The filled block spans x 30..89 and y 40..79 in resized space; mapping
    # back multiplies x by 4 and y by 2, then the unclip expansion pads it.
    assert min(xs) < 30 * 4 < max(xs)
    assert min(ys) < 40 * 2 < max(ys)
    assert all(0 <= x <= 640 for x in xs)
    assert all(0 <= y <= 320 for y in ys)


def test_quads_are_clockwise_from_top_left() -> None:
    prob = _rotated_rect_map(320, 320, [((160.0, 160.0), (200.0, 40.0), 15.0)])

    quads = detect_lines(prob, orig_width=320, orig_height=320, ratio_h=1.0, ratio_w=1.0)

    assert len(quads) == 1
    points = np.asarray(quads[0].points, dtype=np.float64)
    assert len(points) == 4
    start = int(np.argmin(points[:, 0] + points[:, 1]))
    assert start == 0
    area = float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1])
    )
    assert area > 0


def test_unclip_expands_by_area_times_ratio_over_perimeter() -> None:
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])

    expanded = unclip(square, 1.4)

    assert expanded is not None
    # Distance is 100*100*1.4/400 = 35, so the buffered square spans -35..135.
    assert expanded[:, 0].min() <= -33.0
    assert expanded[:, 0].max() >= 133.0
    assert expanded[:, 1].min() <= -33.0
    assert expanded[:, 1].max() >= 133.0


def test_unclip_returns_none_for_degenerate_input() -> None:
    assert unclip(np.zeros((0, 2)), 1.4) is None
    assert unclip(np.array([[1.0, 1.0], [1.0, 1.0]]), 1.4) is None


def _shoelace(points: np.ndarray) -> float:
    corners = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return float(
        np.sum(
            corners[:, 0] * np.roll(corners[:, 1], -1) - np.roll(corners[:, 0], -1) * corners[:, 1]
        )
    )


def _rotated_corners(
    center: tuple[float, float], size: tuple[float, float], angle_deg: float
) -> np.ndarray:
    width, height = size
    theta = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = np.array(
        [
            [width / 2, height / 2],
            [-width / 2, height / 2],
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
        ]
    )
    return local @ rotation.T + np.array(center)


def test_order_quad_clockwise_keeps_all_corners_of_a_steep_diamond() -> None:
    diamond = np.array([[0.0, 10.0], [10.0, 0.0], [20.0, 10.0], [10.0, 20.0]])

    ordered = order_quad_clockwise(diamond)

    assert len({tuple(point) for point in ordered.tolist()}) == 4
    assert sorted(map(tuple, ordered.tolist())) == sorted(map(tuple, diamond.tolist()))
    assert _shoelace(ordered) > 0
    assert tuple(ordered[0]) == (10.0, 0.0)


def test_order_quad_clockwise_handles_a_sixty_degree_rectangle() -> None:
    corners = _rotated_corners((0.0, 0.0), (200.0, 40.0), 60.0)

    ordered = order_quad_clockwise(corners)

    assert len({tuple(point) for point in ordered.tolist()}) == 4
    assert sorted(map(tuple, ordered.tolist())) == sorted(map(tuple, corners.tolist()))
    assert _shoelace(ordered) > 0
    np.testing.assert_allclose(
        ordered[0], _rotated_corners((0.0, 0.0), (200.0, 40.0), 60.0)[2], atol=1e-9
    )


def test_order_quad_clockwise_handles_a_tall_thin_eighty_degree_rectangle() -> None:
    corners = _rotated_corners((0.0, 0.0), (10.0, 100.0), 80.0)

    ordered = order_quad_clockwise(corners)

    assert len({tuple(point) for point in ordered.tolist()}) == 4
    assert sorted(map(tuple, ordered.tolist())) == sorted(map(tuple, corners.tolist()))
    assert _shoelace(ordered) > 0
    np.testing.assert_allclose(
        ordered[0], _rotated_corners((0.0, 0.0), (10.0, 100.0), 80.0)[1], atol=1e-9
    )


def test_detect_lines_keeps_a_fifty_degree_rectangle_whole() -> None:
    prob = _rotated_rect_map(320, 320, [((160.0, 160.0), (200.0, 40.0), 50.0)])

    quads = detect_lines(prob, orig_width=320, orig_height=320, ratio_h=1.0, ratio_w=1.0)

    assert len(quads) == 1
    assert len({tuple(point) for point in quads[0].points}) == 4
    assert _shoelace(np.asarray(quads[0].points)) > 0
