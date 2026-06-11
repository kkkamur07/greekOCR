"""Annotation schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class ModelCharacterConfidence(BaseModel):
    char: str
    probability: float = Field(ge=0.0, le=1.0)


class Segment(BaseModel):
    id: str
    number: int
    kind: Literal["polygon", "rectangle"]
    points: list[list[float]]
    paired_text_line_index: int | None = None
    text_override: str | None = None
    model_transcription: str | None = None
    model_transcription_confidence: list[ModelCharacterConfidence] | None = None
    model_transcription_at: str | None = None
    source: Literal["manual", "kraken"] = "manual"
    kraken_ceiling: list[list[float]] | None = None


class ExportMetadata(BaseModel):
    exported_at: str
    content_hash: str


class PageAnnotation(BaseModel):
    segments: list[Segment] = Field(default_factory=list)
    export_metadata: ExportMetadata | None = None
    locked: bool = False
    binarized_at: str | None = None
