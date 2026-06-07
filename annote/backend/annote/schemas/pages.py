"""Page catalogue schemas."""

from pydantic import BaseModel


class PairingProgress(BaseModel):
    paired_count: int
    unpaired_count: int
    text_line_count: int
    unused_line_count: int


class PageSummary(BaseModel):
    stem: str
    has_transcription: bool
    segment_count: int
    export_dirty: bool
    locked: bool = False
    pairing: PairingProgress


class PageListResponse(BaseModel):
    pages: list[PageSummary]


class TextLineOut(BaseModel):
    index: int
    text: str


class TranscriptionResponse(BaseModel):
    raw_text: str | None
    text_lines: list[TextLineOut]
    status: str = "ok"
