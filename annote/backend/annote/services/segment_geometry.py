"""Geometry helpers for segment layout."""


def segment_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def segment_centroid(points: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def sort_segments_reading_order(segments: list) -> list:
    return sorted(
        segments,
        key=lambda s: (segment_centroid(s.points)[1], segment_centroid(s.points)[0]),
    )
