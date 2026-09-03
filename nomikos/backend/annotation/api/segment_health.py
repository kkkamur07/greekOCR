"""Segment health routes: what is wrong with a page, and the one fix a human picked.

Every apply route takes ids and nothing else. The geometry is recomputed on the
server from the rows as they stand, so a stale tab cannot write an outline
measured against a page somebody else has already changed. Each returns the
part's full line list, the same shape the editor already reloads after a
history restore, so the client never has to merge a partial response into its
own state.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.api.schemas import (
    SegmentDeleteRequest,
    SegmentHealthResponse,
    SegmentMergeRequest,
    SegmentMergeResponse,
    SegmentOverlapResponse,
    SegmentSplitRequest,
    SegmentSplitResponse,
    SegmentSuspectResponse,
    SegmentTrimRequest,
)
from backend.annotation.application.segment_health_service import (
    SegmentHealthReport,
    SegmentHealthService,
)
from backend.document.api.line_responses import line_response
from backend.document.api.schemas import LineResponse
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(
    prefix="/projects/{project_id}/documents/{document_id}/parts/{part_id}/segment-health",
    tags=["segment-health"],
)
_service = SegmentHealthService()


def _report_response(report: SegmentHealthReport) -> SegmentHealthResponse:
    return SegmentHealthResponse(
        part_id=report.part_id,
        page_width=report.page_width,
        page_height=report.page_height,
        measured_page=report.measured_page,
        line_count=report.line_count,
        considered_count=report.considered_count,
        finding_count=report.finding_count,
        suspects=[
            SegmentSuspectResponse(line_id=UUID(item.line_id), reasons=item.reasons)
            for item in report.suspects
        ],
        spanning=[
            SegmentSplitResponse(
                line_id=UUID(item.line_id), cuts=item.cuts, piece_count=len(item.pieces)
            )
            for item in report.spanning
        ],
        fragments=[
            SegmentMergeResponse(
                primary_id=UUID(item.primary_id), fragment_id=UUID(item.fragment_id)
            )
            for item in report.fragments
        ],
        overlaps=[
            SegmentOverlapResponse(
                upper_id=UUID(item.upper_id),
                lower_id=UUID(item.lower_id),
                ratio=item.ratio,
                cut=item.cut,
                upper_loss=item.upper_loss,
                lower_loss=item.lower_loss,
                duplicate=item.duplicate,
            )
            for item in report.overlaps
        ],
    )


@router.get("", response_model=SegmentHealthResponse)
async def get_segment_health(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SegmentHealthResponse:
    report = await _service.report(db, current_user, project_id, document_id, part_id)
    return _report_response(report)


@router.post("/splits", response_model=list[LineResponse])
async def split_spanning_segment(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    payload: SegmentSplitRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LineResponse]:
    lines = await _service.apply_split(
        db, current_user, project_id, document_id, part_id, payload.line_id
    )
    return [line_response(line) for line in lines]


@router.post("/merges", response_model=list[LineResponse])
async def merge_fragment(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    payload: SegmentMergeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LineResponse]:
    lines = await _service.apply_merge(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        payload.primary_id,
        payload.fragment_id,
    )
    return [line_response(line) for line in lines]


@router.post("/trims", response_model=list[LineResponse])
async def trim_overlap(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    payload: SegmentTrimRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LineResponse]:
    lines = await _service.apply_trim(
        db, current_user, project_id, document_id, part_id, payload.upper_id, payload.lower_id
    )
    return [line_response(line) for line in lines]


# POST rather than DELETE: this is "act on the suspect the report offered", not
# "remove this row", and the service refuses any id the report did not flag.
@router.post("/deletions", response_model=list[LineResponse])
async def delete_suspect(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    payload: SegmentDeleteRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LineResponse]:
    lines = await _service.apply_delete(
        db, current_user, project_id, document_id, part_id, payload.line_id
    )
    return [line_response(line) for line in lines]
