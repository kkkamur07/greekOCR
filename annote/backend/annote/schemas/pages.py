"""Page catalogue schemas."""

from pydantic import BaseModel


class PageSummary(BaseModel):
    stem: str
    has_transcription: bool
    segment_count: int
    export_dirty: bool


class PageListResponse(BaseModel):
    pages: list[PageSummary]


class TextLineOut(BaseModel):
    index: int
    text: str


class TranscriptionResponse(BaseModel):
    raw_text: str | None
    text_lines: list[TextLineOut]
    status: str = "ok"
