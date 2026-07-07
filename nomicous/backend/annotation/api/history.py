"""Annotation history routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.api.schemas import (
    AnnotationHistorySnapshotResponse,
    AnnotationHistorySnapshotSummaryResponse,
)
from backend.annotation.application.history_service import AnnotationHistoryService
from backend.document.api.line_responses import line_response
from backend.document.api.schemas import LineResponse
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(
    prefix="/projects/{project_id}/documents/{document_id}/parts/{part_id}/history",
    tags=["annotation-history"],
)
_service = AnnotationHistoryService()


@router.post(
    "",
    response_model=AnnotationHistorySnapshotResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_history_snapshot(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AnnotationHistorySnapshotResponse:
    snapshot = await _service.create_snapshot(db, current_user, project_id, document_id, part_id)
    return AnnotationHistorySnapshotResponse.model_validate(snapshot)


@router.get("", response_model=list[AnnotationHistorySnapshotSummaryResponse])
async def list_history_snapshots(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[AnnotationHistorySnapshotSummaryResponse]:
    snapshots = await _service.list_snapshots(db, current_user, project_id, document_id, part_id)
    return [
        AnnotationHistorySnapshotSummaryResponse.model_validate(snapshot)
        for snapshot in snapshots
    ]


@router.post("/{snapshot_id}/restore", response_model=list[LineResponse])
async def restore_history_snapshot(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    snapshot_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LineResponse]:
    lines = await _service.restore_snapshot(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        snapshot_id,
    )
    return [line_response(line) for line in lines]
