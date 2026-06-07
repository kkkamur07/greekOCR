"""Annotation schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class Segment(BaseModel):
    id: str
    number: int
    kind: Literal["polygon", "rectangle"]
    points: list[list[float]]
    paired_text_line_index: int | None = None
    text_override: str | None = None


class ExportMetadata(BaseModel):
    exported_at: str
    content_hash: str


class PageAnnotation(BaseModel):
    segments: list[Segment] = Field(default_factory=list)
    export_metadata: ExportMetadata | None = None
    locked: bool = False
