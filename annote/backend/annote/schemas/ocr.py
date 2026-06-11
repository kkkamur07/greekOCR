"""OCR prediction streaming event schemas."""

from typing import Literal

from pydantic import BaseModel


class OcrResult(BaseModel):
    processed_count: int


class OcrProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    current: int
    total: int
    segment_number: int
    segment_id: str


class OcrDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    result: OcrResult


class OcrErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    detail: str
