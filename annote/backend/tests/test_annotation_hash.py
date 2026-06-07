"""Annotation content hash — export dirty tracking."""

from annote.schemas.annotation import PageAnnotation, Segment
from annote.services.annotation_store import annotation_content_hash

SEG = Segment(
    id="seg-1",
    number=1,
    kind="rectangle",
    points=[[10, 10], [90, 10], [90, 40], [10, 40]],
    paired_text_line_index=1,
)


def test_content_hash_ignores_locked_flag():
    unlocked = PageAnnotation(segments=[SEG], locked=False)
    locked = PageAnnotation(segments=[SEG], locked=True)

    assert annotation_content_hash(unlocked) == annotation_content_hash(locked)
