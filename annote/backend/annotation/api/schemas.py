from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class AnnotationHistorySnapshotResponse(BaseModel):
    id: UUID
    part_id: UUID
    state: dict
    line_count: int
    paired_line_count: int
    created_at: datetime

    model_config = {"from_attributes": True}
