"""Annotation history routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.api.schemas import AnnotationHistorySnapshotResponse
from backend.annotation.application.history_service import AnnotationHistoryService
from backend.document.api.schemas import LineResponse, LineTranscriptionResponse
from backend.document.infrastructure.orm_models import Line
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(
    prefix="/projects/{project_id}/documents/{document_id}/parts/{part_id}/history",
    tags=["annotation-history"],
)
_service = AnnotationHistoryService()


def _line_response(line: Line) -> LineResponse:
    return LineResponse(
        id=line.id,
        part_id=line.part_id,
        block_id=line.block_id,
        order=line.order,
        baseline=line.baseline,
        mask=line.mask,
        kind=line.kind,
        points=line.points,
        source=line.source,
        source_metadata=line.source_metadata,
        kraken_ceiling=line.kraken_ceiling,
        manual_geometry=line.manual_geometry,
        line_transcriptions=[
            LineTranscriptionResponse(
                id=transcription.id,
                transcription_id=transcription.transcription_id,
                transcription_kind=transcription.transcription.kind,
                text=transcription.text,
                confidence=transcription.confidence,
            )
            for transcription in line.transcriptions
        ],
        created_at=line.created_at,
    )


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


@router.get("", response_model=list[AnnotationHistorySnapshotResponse])
async def list_history_snapshots(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[AnnotationHistorySnapshotResponse]:
    snapshots = await _service.list_snapshots(db, current_user, project_id, document_id, part_id)
    return [AnnotationHistorySnapshotResponse.model_validate(snapshot) for snapshot in snapshots]


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
    return [_line_response(line) for line in lines]
