"""Document-level batch actions: segment or transcribe a whole chapter in one request.

Same prefix as the per-part document routes, and deliberately the same gate: these are
*editing* actions, so project membership is the whole of the authorization. Publishing is
the only thing on a document that asks for ownership, because that is exposure rather
than editing.

The DTOs live in this module rather than in ``schemas.py`` so that the batch surface is
one file to read: the request that names a scope, the response the menu renders, and the
routes that connect them.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.application.document_batch_service import (
    BatchEnqueueResult,
    DocumentBatchService,
    SegmentScope,
    TranscribeScope,
)
from backend.document.infrastructure.document_batch_repository import WorkflowCounts
from backend.jobs.api.schemas import EnqueueJobResponse, enqueue_job_response_from_orm
from backend.ml.application.capacity_service import InferenceCapacityService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])

_batch = DocumentBatchService()
_capacity = InferenceCapacityService()


class SegmentDocumentRequest(BaseModel):
    """Which pages to segment, and with which model.

    The default is the additive scope. ``all`` re-segments every page and discards the
    transcriptions on each one, so a client that wants that has to say the word.
    """

    scope: SegmentScope = SegmentScope.unsegmented
    model_id: UUID | None = None


class TranscribeDocumentRequest(BaseModel):
    """Which segmented pages to transcribe, and with which model."""

    scope: TranscribeScope = TranscribeScope.unpaired
    model_id: UUID | None = None


class DocumentBatchJobsResponse(BaseModel):
    """The jobs a fan-out created, and how many pages it left alone.

    Each job is announced the way a single enqueue announces it: id plus the
    **execution target** fixed at submission. A batch is the same enqueue run
    once per page, so it states the host the same way, at the same moment,
    rather than leaving a whole chapter's worth of jobs to be discovered on the
    next poll.
    """

    jobs: list[EnqueueJobResponse]
    queued: int
    skipped: int


class DocumentWorkflowCountsResponse(BaseModel):
    """Progress through a document, in the numbers the batch menu renders.

    ``unsegmented`` counts pages with no segments at all. ``unpaired`` counts *segmented*
    pages whose segments carry no text yet, so the two never overlap and neither promises
    work the platform would refuse.
    """

    total: int
    reviewed: int
    unsegmented: int
    unpaired: int


def _jobs_response(result: BatchEnqueueResult) -> DocumentBatchJobsResponse:
    return DocumentBatchJobsResponse(
        jobs=[enqueue_job_response_from_orm(job) for job in result.jobs],
        queued=result.queued,
        skipped=result.skipped,
    )


def _counts_response(counts: WorkflowCounts) -> DocumentWorkflowCountsResponse:
    return DocumentWorkflowCountsResponse(
        total=counts.total,
        reviewed=counts.reviewed,
        unsegmented=counts.unsegmented,
        unpaired=counts.unpaired,
    )


@router.post(
    "/{document_id}/jobs/segment",
    response_model=DocumentBatchJobsResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def segment_document(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    body: SegmentDocumentRequest | None = None,
) -> DocumentBatchJobsResponse:
    """Queue segmentation across a document, one job per page.

    ``scope="unsegmented"`` (the default) queues only pages that have no segments yet.
    Nothing already on the document can be lost to it.

    ``scope="all"`` re-segments **every** page, and that destroys work: segmentation
    replaces a page's lines, transcriptions hang off those lines, so every transcription
    on every page of this document is discarded - the model's output and the
    researcher's approved ground truth alike. There is no undo. Ask for it by name, never
    by default, and confirm it with the researcher before sending it.

    Returns 202 with ``queued: 0`` when the scope matches no page, including on a
    document with no pages at all: a batch that had nothing to do has done what was
    asked. Refuses with 409 when no inference host has capacity, before writing any job,
    so a chapter is never left half queued.
    """
    body = body or SegmentDocumentRequest()
    result = await _batch.segment_document(
        db,
        current_user,
        project_id,
        document_id,
        execution=await _capacity.execution_request(db, current_user),
        scope=body.scope,
        model_id=body.model_id,
    )
    return _jobs_response(result)


@router.post(
    "/{document_id}/jobs/transcribe",
    response_model=DocumentBatchJobsResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def transcribe_document(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    body: TranscribeDocumentRequest | None = None,
) -> DocumentBatchJobsResponse:
    """Queue transcription across a document, one job per segmented page.

    ``scope="unpaired"`` (the default) queues only pages whose segments carry no text
    yet. ``scope="all"`` re-transcribes every segmented page: it adds a transcription
    layer rather than replacing the lines, so unlike re-segmentation it destroys nothing,
    but it does spend inference on pages that already have text.

    Neither scope reaches a page with no segments - there would be nothing to transcribe.
    Those pages count as ``skipped``. Same 202-with-zero and same up-front capacity
    refusal as the segment route.
    """
    body = body or TranscribeDocumentRequest()
    result = await _batch.transcribe_document(
        db,
        current_user,
        project_id,
        document_id,
        execution=await _capacity.execution_request(db, current_user),
        scope=body.scope,
        model_id=body.model_id,
    )
    return _jobs_response(result)


@router.get("/{document_id}/workflow-counts", response_model=DocumentWorkflowCountsResponse)
async def document_workflow_counts(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentWorkflowCountsResponse:
    """How many pages this document has, and how many still need each step.

    These numbers go straight into menu labels next to the actions above, so they are
    counted by Postgres over the whole document in one statement and must match what the
    corresponding scope would queue.
    """
    counts = await _batch.workflow_counts(db, current_user, project_id, document_id)
    return _counts_response(counts)
