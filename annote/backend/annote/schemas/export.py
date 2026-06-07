"""Export request and streaming event schemas."""

from typing import Literal

from pydantic import BaseModel


class ExportWarnings(BaseModel):
    unpaired_segments: list[int]
    unused_text_lines: list[int]


class ExportResponse(BaseModel):
    exported_count: int
    warnings: ExportWarnings
    steps: list[str]


class ExportProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    current: int
    total: int
    segment_number: int
    step: Literal["rectify", "save"]


class ExportDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    result: ExportResponse


class ExportErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    detail: str
