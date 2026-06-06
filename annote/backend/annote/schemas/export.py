"""Export request and streaming event schemas."""

from typing import Literal

from pydantic import BaseModel

from annote.schemas.warnings import ExportResponse


class ExportRequest(BaseModel):
    binarize: bool = False


class ExportProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    current: int
    total: int
    segment_number: int
    step: Literal["rectify", "binarize", "save"]


class ExportDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    result: ExportResponse


class ExportErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    detail: str
