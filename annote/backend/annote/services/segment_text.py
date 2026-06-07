"""Resolve segment text and pairing progress."""

from annote.schemas.annotation import Segment
from annote.schemas.pages import PairingProgress
from annote.services.text_lines import TextLine


def segment_text(segment: Segment, text_lines: list[TextLine]) -> str | None:
    """Return exportable text for a segment, or None if unpaired."""
    if segment.text_override is not None:
        return segment.text_override
    if segment.paired_text_line_index is None:
        return None
    for line in text_lines:
        if line.index == segment.paired_text_line_index:
            return line.text
    return None


def segment_is_paired(segment: Segment) -> bool:
    """True when the segment has a pairing or typed text override."""
    return segment.text_override is not None or segment.paired_text_line_index is not None


def compute_pairing_progress(
    segments: list[Segment],
    text_lines: list[TextLine],
) -> PairingProgress:
    paired_count = sum(1 for s in segments if segment_is_paired(s))
    unpaired_count = len(segments) - paired_count
    used_indices = {
        s.paired_text_line_index
        for s in segments
        if s.paired_text_line_index is not None
    }
    unused_line_count = sum(1 for line in text_lines if line.index not in used_indices)
    return PairingProgress(
        paired_count=paired_count,
        unpaired_count=unpaired_count,
        text_line_count=len(text_lines),
        unused_line_count=unused_line_count,
    )
