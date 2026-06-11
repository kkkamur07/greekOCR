"""Merge incoming annotation saves with existing on-disk state."""

from annote.schemas.annotation import PageAnnotation


def clear_stale_model_transcriptions(existing: PageAnnotation, incoming: PageAnnotation) -> PageAnnotation:
    """Clear model OCR fields on segments whose geometry changed since last save."""
    prev_by_id = {s.id: s for s in existing.segments}
    segments = []
    for seg in incoming.segments:
        prev = prev_by_id.get(seg.id)
        if prev is not None and prev.points != seg.points:
            seg = seg.model_copy(update={"model_transcription": None, "model_transcription_at": None})
        segments.append(seg)
    return incoming.model_copy(update={"segments": segments})
