"""Export warning and response schemas."""

from pydantic import BaseModel


class ExportWarnings(BaseModel):
    unpaired_segments: list[int]
    unused_text_lines: list[int]


class ExportResponse(BaseModel):
    exported_count: int
    warnings: ExportWarnings
    steps: list[str]
