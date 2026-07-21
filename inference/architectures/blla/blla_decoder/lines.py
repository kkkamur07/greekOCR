"""BLLA baseline vectorization and reading order."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
from shapely import geometry as geom
from shapely.ops import nearest_points, unary_union
from skimage import filters
from skimage.graph import MCP_Connect
from skimage.measure import approximate_polygon, label, regionprops
from skimage.morphology import skeletonize


class _LineMCP(MCP_Connect):
    """Connect skeleton endpoints using the reference baseline scoring rule."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.connections: dict[tuple[int, int], tuple[object, object, float]] = {}
        self.scores: defaultdict[tuple[int, int], float] = defaultdict(lambda: np.inf)

    def create_connection(
        self,
        id1: int,
        id2: int,
        pos1: object,
        pos2: object,
        cost1: float,
        cost2: float,
    ) -> None:
        key = (min(id1, id2), max(id1, id2))
        score = cost1 + cost2
        if self.scores[key] > score:
            self.connections[key] = (pos1, pos2, score)
            self.scores[key] = score

    def get_connections(self) -> list[np.ndarray]:
        return [
            np.concatenate([self.traceback(pos1), self.traceback(pos2)[::-1]])
            for pos1, pos2, _score in self.connections.values()
        ]

    def goal_reached(self, _int_index: int, float_cumcost: float) -> int:
        return 2 if float_cumcost else 0


def _moore_neighborhood(current: np.ndarray, backtrack: np.ndarray) -> np.ndarray:
    operations = np.array(
        [
            [-1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1],
        ]
    )
    neighbors = (current + operations).astype(int)
    for index, point in enumerate(neighbors):
        if np.all(point == backtrack):
            return np.concatenate((neighbors[index:], neighbors[:index]))
    return np.empty((0, 2), dtype=int)


def _boundary_tracing(region: object) -> np.ndarray:
    coords = region.coords  # type: ignore[attr-defined]
    mins = np.amin(coords, axis=0)
    maxs = np.amax(coords, axis=0)
    binary = np.zeros((maxs[0] - mins[0] + 3, maxs[1] - mins[1] + 3))
    y = coords[:, 0] - mins[0] + 1
    x = coords[:, 1] - mins[1] + 1
    binary[y, x] = 1

    index = 0
    while True:
        start = [y[index], x[index]]
        focus_start = binary[start[0] - 1 : start[0] + 2, start[1] - 1 : start[1] + 2]
        if np.sum(focus_start) > 1:
            break
        index += 1

    if binary[start[0] + 1, start[1]] == 0 and binary[start[0] + 1, start[1] - 1] == 0:
        backtrack_start = [start[0] + 1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = np.asarray(start)
    backtrack = np.asarray(backtrack_start)
    boundary: list[np.ndarray] = []
    while True:
        neighbors = _moore_neighborhood(current, backtrack)
        values = binary[neighbors[:, 0], neighbors[:, 1]]
        index = int(np.argmax(values))
        boundary.append(current)
        backtrack = neighbors[index - 1]
        current = neighbors[index]
        if np.all(current == start) and np.all(backtrack == backtrack_start):
            break
    return np.asarray(boundary) + [mins[0] - 1, mins[1] - 1]


def _extend_boundaries(
    baselines: list[list[list[int]]],
    baseline_map: np.ndarray,
) -> list[list[list[int]]]:
    labelled = label(baseline_map)
    boundaries: list[object] = []
    for region in regionprops(labelled):
        if region.area < 6:
            continue
        try:
            boundary = _boundary_tracing(region)
            if len(boundary) > 3:
                boundaries.append(geom.Polygon(boundary).simplify(0.01).buffer(0))
        except Exception:
            continue

    for baseline in baselines:
        line = geom.LineString(baseline)
        try:
            boundary_polygon = next(
                candidate for candidate in boundaries if candidate.contains(line)
            )
        except StopIteration:
            continue

        if boundary_polygon.contains(geom.Point(baseline[0])):
            intersection = boundary_polygon.boundary.intersection(
                geom.LineString(
                    [
                        (
                            baseline[0][0] - 10 * (baseline[1][0] - baseline[0][0]),
                            baseline[0][1] - 10 * (baseline[1][1] - baseline[0][1]),
                        ),
                        baseline[0],
                    ]
                )
            )
            if intersection.geom_type != "Point":
                baseline[0] = np.asarray(
                    nearest_points(geom.Point(baseline[0]), boundary_polygon)[1].coords[0],
                    dtype=int,
                ).tolist()
            else:
                baseline[0] = np.asarray(intersection.coords[0], dtype=int).tolist()

        if boundary_polygon.contains(geom.Point(baseline[-1])):
            intersection = boundary_polygon.boundary.intersection(
                geom.LineString(
                    [
                        (
                            baseline[-1][0] - 10 * (baseline[-2][0] - baseline[-1][0]),
                            baseline[-1][1] - 10 * (baseline[-2][1] - baseline[-1][1]),
                        ),
                        baseline[-1],
                    ]
                )
            )
            if intersection.geom_type != "Point":
                baseline[-1] = np.asarray(
                    nearest_points(geom.Point(baseline[-1]), boundary_polygon)[1].coords[0],
                    dtype=int,
                ).tolist()
            else:
                baseline[-1] = np.asarray(intersection.coords[0], dtype=int).tolist()
    return baselines


def vectorize_lines(
    channels: np.ndarray,
    *,
    threshold: float,
    min_length: float,
    text_direction: str = "horizontal",
    max_endpoints: int = 400,
) -> list[list[list[int]]]:
    """Apply baseline skeletonization and endpoint connection."""

    if text_direction not in {"horizontal", "vertical"}:
        raise ValueError(f"Invalid text direction {text_direction!r}")

    start_map, end_map, baseline_map = channels[:3]
    filtered_baseline = filters.sato(baseline_map, black_ridges=False, mode="constant")
    binary_baseline = filtered_baseline > threshold
    line_skeleton = skeletonize(binary_baseline)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    extrema = np.transpose(
        np.where((convolve2d(line_skeleton, kernel, mode="same") == 11) * line_skeleton)
    )

    if len(extrema) > max_endpoints:
        skeleton_labels = label(line_skeleton)
        extrema_components = skeleton_labels[extrema[:, 0], extrema[:, 1]]
        component_ids, component_counts = np.unique(extrema_components, return_counts=True)
        endpoint_counts = dict(zip(component_ids.tolist(), component_counts.tolist()))
        valid_components = {
            component_id for component_id, count in endpoint_counts.items() if count <= 10
        }
        remaining_endpoints = sum(
            endpoint_counts[component_id] for component_id in valid_components
        )
        if remaining_endpoints > max_endpoints:
            component_sizes = np.bincount(skeleton_labels.ravel())
            sorted_components = sorted(
                valid_components,
                key=lambda component_id: component_sizes[component_id],
                reverse=True,
            )
            budget_components: set[int] = set()
            budget = 0
            for component_id in sorted_components:
                endpoints = endpoint_counts[component_id]
                if budget + endpoints > max_endpoints:
                    break
                budget_components.add(component_id)
                budget += endpoints
            valid_components = budget_components
        extrema = extrema[np.isin(extrema_components, list(valid_components))]

    if len(extrema) < 2:
        return []

    mcp = _LineMCP(~line_skeleton)
    try:
        mcp.find_costs(extrema)
    except ValueError:
        return []

    lines = [approximate_polygon(line, 3).tolist() for line in mcp.get_connections()]
    lines = _extend_boundaries(lines, binary_baseline)
    filtered_start = maximum_filter(start_map, size=20)
    filtered_end = maximum_filter(end_map, size=20)

    oriented: list[list[list[int]]] = []
    for baseline in lines:
        first = tuple(baseline[0])
        last = tuple(baseline[-1])
        if filtered_start[first] - filtered_end[first] > 0.2 and (
            filtered_start[last] - filtered_end[last] < -0.2
        ):
            pass
        elif filtered_start[first] - filtered_end[first] < -0.2 and (
            filtered_start[last] - filtered_end[last] > 0.2
        ):
            baseline = baseline[::-1]
        elif text_direction == "horizontal":
            if baseline[0][1] > baseline[-1][1]:
                baseline = baseline[::-1]
        elif baseline[0][0] > baseline[-1][0]:
            baseline = baseline[::-1]
        if geom.LineString(baseline).length >= min_length:
            oriented.append([point[::-1] for point in baseline])
    return oriented


def vectorize_regions(region_probability: np.ndarray) -> list[list[list[int]]]:
    """Vectorize text regions for reading order and ROIs."""

    boundaries = []
    for region in regionprops(label(region_probability > 0.5)):
        boundary = _boundary_tracing(region)
        if len(boundary) > 2:
            boundaries.append(geom.Polygon(boundary))
    merged = unary_union(boundaries)
    if merged.geom_type == "Polygon":
        polygons = [merged.boundary.simplify(10)]
    else:
        polygons = [item.boundary.simplify(10) for item in merged.geoms]
    return [np.asarray(polygon.coords, dtype=np.uint)[:, [1, 0]].tolist() for polygon in polygons]


def is_in_region(line: geom.LineString, region: geom.Polygon) -> bool:
    midpoint = line.interpolate(0.5, normalized=True)
    return region.contains(midpoint)


def _reading_order(
    lines: list[tuple[slice, slice]],
    *,
    text_direction: str = "lr",
) -> np.ndarray:
    order = np.zeros((len(lines), len(lines)), dtype="uint8")

    def x_overlaps(first: tuple[slice, slice], second: tuple[slice, slice]) -> bool:
        return first[1].start < second[1].stop and first[1].stop > second[1].start

    def above(first: tuple[slice, slice], second: tuple[slice, slice]) -> bool:
        return first[0].start < second[0].start

    def left_of(first: tuple[slice, slice], second: tuple[slice, slice]) -> bool:
        return first[1].stop < second[1].start

    def separates(
        middle: tuple[slice, slice],
        first: tuple[slice, slice],
        second: tuple[slice, slice],
    ) -> bool:
        if middle == first or middle == second:
            return False
        if middle[0].stop < min(first[0].start, second[0].start):
            return False
        if middle[0].start > max(first[0].stop, second[0].stop):
            return False
        return middle[1].start < first[1].stop and middle[1].stop > second[1].start

    horizontal_order = (
        (lambda first, second: not left_of(first, second)) if text_direction == "rl" else left_of
    )
    for first_index, first in enumerate(lines):
        for second_index, second in enumerate(lines):
            if x_overlaps(first, second):
                if above(first, second):
                    order[first_index, second_index] = 1
            elif not any(separates(middle, first, second) for middle in lines) and horizontal_order(
                first, second
            ):
                order[first_index, second_index] = 1
    return order


def _topological_sort(order: np.ndarray) -> list[int]:
    visited = np.zeros(len(order))
    result: list[int] = []

    def visit(index: int) -> None:
        if visited[index]:
            return
        visited[index] = 1
        predecessors = np.nonzero(np.ravel(order[:, index]))[0]
        for predecessor in predecessors:
            visit(int(predecessor))
        result.append(index)

    for index in range(len(order)):
        visit(index)
    return result


def reading_order_indices(
    baselines: list[list[list[int]]],
    regions: list[list[list[int]]] | None = None,
) -> list[int]:
    """Return polygon-aware reading-order indices."""

    if not regions:
        slices = [
            (
                slice(geom.LineString(line).bounds[1], geom.LineString(line).bounds[3]),
                slice(geom.LineString(line).bounds[0], geom.LineString(line).bounds[2]),
            )
            for line in baselines
        ]
        return _topological_sort(_reading_order(slices))

    region_polygons = [geom.Polygon(region) for region in regions]
    region_lines: list[list[tuple[int, tuple[slice, slice]]]] = [[] for _ in region_polygons]
    bounds: list[tuple[slice, slice]] = []
    indices: dict[int, tuple[str, object]] = {}
    for line_index, baseline in enumerate(baselines):
        line = geom.LineString(baseline)
        line_slices = (
            slice(line.bounds[1], line.bounds[3]),
            slice(line.bounds[0], line.bounds[2]),
        )
        for region_index, region in enumerate(region_polygons):
            if is_in_region(line, region):
                region_lines[region_index].append((line_index, line_slices))
                break
        else:
            bounds.append(line_slices)
            indices[line_index] = ("line", baseline)

    intra_region_order: list[list[int]] = [[] for _ in region_polygons]
    last_line_index = len(baselines) - 1
    for region_index, region in enumerate(region_polygons):
        if region_lines[region_index]:
            region_order = _topological_sort(
                _reading_order([item[1] for item in region_lines[region_index]])
            )
            intra_region_order[region_index] = [
                region_lines[region_index][position][0] for position in region_order
            ]
            region_bounds = region.bounds
            bounds.append(
                (
                    slice(region_bounds[1], region_bounds[3]),
                    slice(region_bounds[0], region_bounds[2]),
                )
            )
            indices[last_line_index + region_index + 1] = ("region", region_index)

    order = _topological_sort(_reading_order(bounds))
    sorted_indices = sorted(indices)
    sorted_indices = [sorted_indices[position] for position in order]
    ordered: list[int] = []
    for index in sorted_indices:
        kind, value = indices[index]
        if kind == "line":
            ordered.append(index)
        else:
            ordered.extend(intra_region_order[int(value)])
    return ordered


def apply_reading_order(
    baselines: list[list[list[int]]],
    regions: list[list[list[int]]] | None = None,
) -> list[list[list[int]]]:
    """Apply the default polygon-aware reading order to baselines."""

    indices = reading_order_indices(baselines, regions)
    return [baselines[index] for index in indices]
