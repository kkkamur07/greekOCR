"""Annotation history schemas."""

from pydantic import BaseModel

from annote.schemas.annotation import PageAnnotation


class HistorySnapshotSummary(BaseModel):
    id: str
    timestamp: str
    reason: str
    pairing_progress_percent: int


class HistoryListResponse(BaseModel):
    snapshots: list[HistorySnapshotSummary]


class HistorySnapshotRecord(BaseModel):
    id: str
    timestamp: str
    reason: str
    pairing_progress_percent: int
    protected: bool
    annotation: PageAnnotation
