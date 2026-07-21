"""BLLA polygonal environment extraction."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_cdt
from shapely import geometry as geom
from shapely.ops import nearest_points, unary_union
from shapely.validation import explain_validity
from skimage import draw
from skimage.transform import AffineTransform, warp


def _ray_intersect_boundaries(
    ray: np.ndarray,
    direction: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    direction_fraction = np.empty(2, dtype=ray.dtype)
    direction_fraction[direction == 0.0] = np.inf
    direction_fraction[direction != 0.0] = np.divide(1.0, direction[direction != 0.0])
    t1 = (-ray[0]) * direction_fraction[0]
    t2 = (bounds[0] - ray[0]) * direction_fraction[0]
    t3 = (-ray[1]) * direction_fraction[1]
    t4 = (bounds[1] - ray[1]) * direction_fraction[1]
    t_min = max(min(t1, t2), min(t3, t4))
    t_max = min(max(t1, t2), max(t3, t4))
    distance = min(value for value in (t_min, t_max) if value >= 0)
    return ray + direction * distance


def _make_polygonal_mask(polygon: np.ndarray, shape: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", shape, 0)
    ImageDraw.Draw(mask).polygon(
        [tuple(point) for point in polygon.astype(int).tolist()],
        fill=255,
        width=2,
    )
    return mask


def _rotate_array(
    image: np.ndarray,
    angle: float,
    scale: float,
) -> tuple[AffineTransform, np.ndarray]:
    rows, cols = image.shape[:2]
    transform = AffineTransform(rotation=angle, scale=(1 / scale, 1))
    corners = np.array([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]])
    corners = transform.inverse(corners)
    min_col, min_row = corners[:, 0].min(), corners[:, 1].min()
    max_col, max_row = corners[:, 0].max(), corners[:, 1].max()
    output_shape = tuple(
        int(value) for value in np.around((max_row - min_row + 1, max_col - min_col + 1))
    )
    translation = transform([[min_col, min_row]])
    transform = AffineTransform(
        rotation=angle,
        scale=(1 / scale, 1),
        translation=translation.flatten().tolist(),
    )
    return transform, warp(
        image,
        transform,
        output_shape=output_shape,
        order=0,
        cval=99999,
        clip=False,
        preserve_range=True,
    )


def _calc_seam(
    baseline: np.ndarray,
    polygon: np.ndarray,
    angle: float,
    image_features: np.ndarray,
    bias: int = 150,
) -> np.ndarray:
    mask_value = 99999
    c_min, c_max = int(polygon[:, 0].min()), int(polygon[:, 0].max())
    r_min, r_max = int(polygon[:, 1].min()), int(polygon[:, 1].max())
    patch = image_features[r_min : r_max + 2, c_min : c_max + 2].copy()
    mask = np.ones_like(patch)
    for line_start, line_end in zip(
        baseline[:-1] - (c_min, r_min),
        baseline[1:] - (c_min, r_min),
    ):
        line_locations = draw.line(
            line_start[1],
            line_start[0],
            line_end[1],
            line_end[0],
        )
        mask[line_locations] = 0
    distance_bias = distance_transform_cdt(mask)
    polygon_mask = (
        np.asarray(_make_polygonal_mask(polygon - (c_min, r_min), patch.shape[::-1])) <= 128
    )
    polygon_mask = binary_erosion(polygon_mask, border_value=True, iterations=2)
    patch[polygon_mask] = mask_value
    patch += distance_bias * (np.mean(patch[patch != mask_value]) / bias)
    extrema = baseline[(0, -1), :] - (c_min, r_min)
    scale = min(1.0, 600 / (c_max - c_min))
    transform, rotated_patch = _rotate_array(patch, angle, scale)
    x_offsets = np.sort(np.around(transform.inverse(extrema)[:, 0]).astype(int))
    rotated_patch = rotated_patch[:, x_offsets[0] : x_offsets[1] + 1]
    rotated_patch = np.pad(
        rotated_patch,
        ((1, 1), (0, 0)),
        mode="constant",
        constant_values=np.inf,
    )
    rows, cols = rotated_patch.shape
    view = np.lib.stride_tricks.as_strided(
        rotated_patch,
        (cols, rows - 2, 3),
        (rotated_patch.strides[1], rotated_patch.strides[0], rotated_patch.strides[0]),
    )
    costs = rotated_patch[1:-1, 1:].swapaxes(0, 1)
    backtrack = np.zeros_like(costs, dtype=int)
    temporary = np.empty((costs.shape[1]), dtype=np.float32)
    previous = np.arange(-1, len(temporary) - 1)
    for index in np.arange(cols - 1):
        view[index].min(1, temporary)
        backtrack[index] = view[index].argmin(1) + previous
        costs[index] += temporary
    seam: list[tuple[int, int]] = []
    column = int(np.argmin(rotated_patch[1:-1, -1]))
    for index in range(cols - 2, -2, -1):
        seam.append((index + x_offsets[0] + 1, column))
        column = backtrack[index, column]
    seam_array = np.asarray(seam)[::-1]
    seam_mean, seam_std = seam_array[:, 1].mean(), seam_array[:, 1].std()
    seam_array[:, 1] = np.clip(
        seam_array[:, 1],
        seam_mean - seam_std,
        seam_mean + seam_std,
    )
    seam_array = transform(seam_array).astype(int)
    seam_array = seam_array[seam_array.min(axis=1) >= 0]
    in_bounds = (seam_array < polygon_mask.shape[::-1]).T
    seam_array = seam_array[np.logical_and(in_bounds[0], in_bounds[1])]
    seam_array = seam_array[~polygon_mask[seam_array[:, 1], seam_array[:, 0]]]
    return seam_array + (c_min, r_min)


def _extract_polygon(
    env_up: np.ndarray,
    env_bottom: np.ndarray,
    baseline: np.ndarray,
    offset_baseline: np.ndarray,
    end_points: tuple[list[int], list[int]],
    direction: np.ndarray,
    topline: bool,
    offset: int,
    image_features: np.ndarray,
) -> np.ndarray:
    upper_polygon = np.concatenate((baseline, env_up[::-1]))
    bottom_polygon = np.concatenate((baseline, env_bottom[::-1]))
    upper_offset_polygon = np.concatenate((offset_baseline, env_up[::-1]))
    bottom_offset_polygon = np.concatenate((offset_baseline, env_bottom[::-1]))
    angle = np.arctan2(direction[1], direction[0])
    roi_polygon = unary_union([geom.Polygon(upper_polygon), geom.Polygon(bottom_polygon)])
    if topline:
        upper_seam = _calc_seam(baseline, upper_polygon, angle, image_features)
        bottom_seam = _calc_seam(offset_baseline, bottom_offset_polygon, angle, image_features)
    else:
        upper_seam = _calc_seam(offset_baseline, upper_offset_polygon, angle, image_features)
        bottom_seam = _calc_seam(baseline, bottom_polygon, angle, image_features)
    upper_line = geom.LineString(upper_seam).simplify(5)
    bottom_line = geom.LineString(bottom_seam).simplify(5)
    upper_offset = upper_line.parallel_offset(offset // 2, side="right")
    bottom_offset = bottom_line.parallel_offset(offset // 2, side="left")
    if upper_offset.geom_type == "MultiLineString" or offset == 0:
        upper_line = np.asarray(upper_line.coords, dtype=int)
    else:
        upper_line = np.asarray(upper_offset.coords, dtype=int)[::-1]
    if bottom_offset.geom_type == "MultiLineString" or offset == 0:
        bottom_line = np.asarray(bottom_line.coords, dtype=int)
    else:
        bottom_line = np.asarray(bottom_offset.coords, dtype=int)
    polygon = geom.Polygon(
        np.concatenate(([end_points[0]], upper_line, [end_points[-1]], bottom_line[::-1]))
    )
    if not polygon.is_valid:
        polygon = geom.Polygon(
            np.concatenate(([end_points[-1]], upper_line, [end_points[0]], bottom_line))
        )
    if not polygon.is_valid:
        raise ValueError(f"Invalid bounding polygon computed: {explain_validity(polygon)}")
    return np.asarray(roi_polygon.intersection(polygon).boundary.coords, dtype=int)


def _calc_roi(
    line: np.ndarray,
    bounds: np.ndarray,
    supplementary_objects: list[list[list[int]]],
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    line_string = geom.LineString(line)
    interpolated = [line[0]]
    distance = 10
    while distance < line_string.length:
        interpolated.append(np.asarray(line_string.interpolate(distance).coords[0]))
        distance += 10
    interpolated.append(line[-1])
    interpolated_array = np.asarray(interpolated)
    upper_intersections = [
        _ray_intersect_boundaries(
            point,
            (direction * (-1, 1))[::-1],
            bounds + 1,
        ).astype(int)
        for point in interpolated_array
    ]
    bottom_intersections = [
        _ray_intersect_boundaries(
            point,
            (direction * (1, -1))[::-1],
            bounds + 1,
        ).astype(int)
        for point in interpolated_array
    ]
    upper_polygon = geom.Polygon(interpolated_array.tolist() + upper_intersections)
    bottom_polygon = geom.Polygon(interpolated_array.tolist() + bottom_intersections)
    side_a = [geom.LineString(upper_intersections)]
    side_b = [geom.LineString(bottom_intersections)]
    for adjacent in supplementary_objects:
        adjacent_line = geom.LineString(adjacent)
        if upper_polygon.intersects(adjacent_line):
            side_a.append(adjacent_line)
        elif bottom_polygon.intersects(adjacent_line):
            side_b.append(adjacent_line)
    side_a = unary_union(side_a).buffer(1).boundary
    side_b = unary_union(side_b).buffer(1).boundary

    def closest(point: np.ndarray, intersections: object) -> object:
        source = geom.Point(point)
        if intersections.is_empty:
            raise ValueError(f"No intersection with boundaries: {intersections.wkt}")
        if intersections.geom_type == "MultiPoint":
            return min(intersections.geoms, key=lambda item: source.distance(item))
        if intersections.geom_type == "Point":
            return intersections
        if intersections.geom_type == "GeometryCollection" and len(intersections.geoms) > 0:
            candidate = min(intersections.geoms, key=lambda item: source.distance(item))
            return (
                candidate
                if candidate.geom_type == "Point"
                else nearest_points(source, candidate)[1]
            )
        raise ValueError(f"No intersection with boundaries: {intersections.wkt}")

    env_up = []
    env_bottom = []
    for point, upper, bottom in zip(
        interpolated_array,
        upper_intersections,
        bottom_intersections,
    ):
        env_up.append(
            closest(point, geom.LineString([point, upper]).intersection(side_a)).coords[0]
        )
        env_bottom.append(
            closest(point, geom.LineString([point, bottom]).intersection(side_b)).coords[0]
        )
    return np.asarray(env_up, dtype="uint"), np.asarray(env_bottom, dtype="uint")


def calculate_polygonal_environment(
    *,
    baseline: list[list[int]],
    supplementary_objects: list[list[list[int]]],
    image_features: np.ndarray,
    bounds: np.ndarray,
    topline: bool = False,
) -> list[list[int]]:
    """Extract one polygonal line environment."""

    line = geom.LineString(baseline)
    offset = 8 if topline is not None else 0
    offset_line = line.parallel_offset(offset, side="left" if topline else "right")
    line_array = np.asarray(line.coords, dtype=float)
    offset_array = np.asarray(offset_line.coords, dtype=float)
    lengths = np.linalg.norm(np.diff(line_array.T), axis=0)
    direction = np.mean(
        np.diff(line_array.T) * lengths / lengths.sum(),
        axis=1,
    )
    direction = direction / np.sqrt(np.sum(direction**2))
    env_up, env_bottom = _calc_roi(
        line_array,
        bounds,
        supplementary_objects,
        direction,
    )
    return _extract_polygon(
        env_up,
        env_bottom,
        line_array.astype(int),
        offset_array.astype(int),
        (baseline[0], baseline[-1]),
        direction,
        topline,
        offset,
        image_features,
    ).tolist()
