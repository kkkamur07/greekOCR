"""Segment text resolution — shared export and PDF behavior."""

from annote.schemas.annotation import Segment
from annote.services.segment_text import segment_text
from annote.services.text_lines import split_text_lines


def test_segment_text_prefers_text_override_over_paired_line():
    lines = split_text_lines("from file\n")
    segment = Segment(
        id="seg-1",
        number=1,
        kind="rectangle",
        points=[[0, 0], [1, 0], [1, 1], [0, 1]],
        paired_text_line_index=1,
        text_override="override wins",
    )

    assert segment_text(segment, lines) == "override wins"
