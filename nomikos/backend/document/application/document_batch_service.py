"""Document-level fan-out: run one page action across a whole chapter.

Segmenting a 200 page manuscript by opening 200 pages and pressing the same button is
the workflow this replaces. The unit of work does not change: one job is still one page,
claimed by one agent, merged by the same handler. All that is added here is *which pages*
the fan-out covers, and that question is answered in SQL by
:class:`DocumentBatchRepository` rather than by loading the document into Python.

Jobs are created by :class:`DocumentJobEnqueueService` and by nothing else. That service
owns model resolution, execution target, and the shape of the ``jobs`` row; a second
insert path here would be a second set of those rules to keep in step, and they would
drift silently because both would keep producing rows that look right.

**Scope is the safety mechanism, so it is never inferred.** ``segment`` defaults to
``unsegmented`` and ``transcribe`` to ``unpaired``: the scopes that only add work.
``all`` has to be asked for by name because re-segmenting a page deletes its lines, and
the transcriptions hang off those lines - approved text included. See
:meth:`segment_document` for what that costs.

**Capacity is checked once, before the first job.** The per-part service refuses a job
when no eligible inference host has capacity (ADR 0002), but it refuses from inside the
loop, after earlier pages have already been committed. Reading the same condition up
front is what keeps the ordinary failure (nobody is running the agent) from leaving half
a chapter queued.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError
from backend.document.application.document_access import DocumentAccess
from backend.document.application.document_job_enqueue import DocumentJobEnqueueService
from backend.document.infrastructure.document_batch_repository import (
    DocumentBatchRepository,
    WorkflowCounts,
)
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.jobs.infrastructure.orm_models import Job, JobType
from backend.ml.domain.execution import NO_CAPACITY_MESSAGE, ExecutionRequest
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User


class SegmentScope(StrEnum):
    """Which pages a document-level segment run covers.

    ``unsegmented`` is additive: it touches only pages that have never been segmented, so
    nothing a researcher has done can be lost to it. ``all`` re-segments every page and
    is destructive - it is spelled out rather than defaulted to for that reason.
    """

    unsegmented = "unsegmented"
    all = "all"


class TranscribeScope(StrEnum):
    """Which pages a document-level transcribe run covers.

    ``unpaired`` covers pages whose segments carry no text yet. ``all`` re-transcribes
    every segmented page, which adds a transcription layer rather than removing one, so
    it is far less costly than its segment counterpart - but it still spends inference on
    work already done, so it is also named explicitly.
    """

    unpaired = "unpaired"
    all = "all"


@dataclass(frozen=True)
class BatchEnqueueResult:
    """What a fan-out did: the jobs it created, and how many pages it left alone.

    ``skipped`` counts every part of the document that did not get a job, whatever the
    reason - out of scope, already segmented, no segments to transcribe, or a job for the
    same task still in flight. One number rather than a breakdown, because the caller
    renders it as "queued 12, skipped 6" and a page skipped for one reason is, to the
    researcher, the same page skipped.
    """

    jobs: list[Job]
    skipped: int

    @property
    def queued(self) -> int:
        return len(self.jobs)


class DocumentBatchService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        access: DocumentAccess | None = None,
        enqueue: DocumentJobEnqueueService | None = None,
        batch: DocumentBatchRepository | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)
        self._enqueue = enqueue or DocumentJobEnqueueService(
            documents=self._documents, projects=self._projects, access=self._access
        )
        self._batch = batch or DocumentBatchRepository()

    async def workflow_counts(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> WorkflowCounts:
        """How far this document has got. Membership is the whole of the gate."""
        context = await self._access.require_document(session, user, project_id, document_id)
        return await self._batch.workflow_counts(session, context.document.id)

    async def segment_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        execution: ExecutionRequest,
        scope: SegmentScope = SegmentScope.unsegmented,
        model_id: UUID | None = None,
    ) -> BatchEnqueueResult:
        """Queue a segment job per page in ``scope``.

        ``SegmentScope.all`` destroys work. Segmentation replaces a page's lines, and a
        line is what a transcription is attached to, so re-segmenting a page discards
        every transcription on it - the model's output and the researcher's approved
        ground truth alike, since only geometry edits are tracked as manual. There is no
        undo. That is why the default is ``unsegmented`` and why the destructive scope
        has to be named in the request body.
        """
        context = await self._access.require_document(session, user, project_id, document_id)
        document = context.document
        candidates = await self._batch.part_ids_to_segment(
            session, document.id, only_unsegmented=scope is SegmentScope.unsegmented
        )
        return await self._fan_out(
            session,
            user,
            project_id,
            document_id,
            candidates=candidates,
            total_parts=len(document.parts),
            job_type=JobType.segment,
            execution=execution,
            model_id=model_id,
        )

    async def transcribe_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        execution: ExecutionRequest,
        scope: TranscribeScope = TranscribeScope.unpaired,
        model_id: UUID | None = None,
    ) -> BatchEnqueueResult:
        """Queue a transcribe job per segmented page in ``scope``.

        Neither scope reaches an unsegmented page: there would be no lines to run the
        model over, and the per-part service refuses such a job outright. Those pages are
        counted as skipped, which is also how ``workflow-counts`` reports them.
        """
        context = await self._access.require_document(session, user, project_id, document_id)
        document = context.document
        candidates = await self._batch.part_ids_to_transcribe(
            session, document.id, only_unpaired=scope is TranscribeScope.unpaired
        )
        return await self._fan_out(
            session,
            user,
            project_id,
            document_id,
            candidates=candidates,
            total_parts=len(document.parts),
            job_type=JobType.transcribe,
            execution=execution,
            model_id=model_id,
        )

    async def _fan_out(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        candidates: list[UUID],
        total_parts: int,
        job_type: JobType,
        execution: ExecutionRequest,
        model_id: UUID | None,
    ) -> BatchEnqueueResult:
        """Turn an eligible set into jobs, one page at a time, in reading order."""
        in_flight = await self._batch.part_ids_with_job_in_flight(
            session, document_id, job_type=job_type
        )
        eligible = [part_id for part_id in candidates if part_id not in in_flight]
        if not eligible:
            # Nothing to do is a completed request, not a failure: "segment the pages
            # that need it" against a chapter that is already segmented has done exactly
            # what was asked. Returning 202 with queued 0 also keeps the empty document
            # and the fully-processed document on the same path.
            return BatchEnqueueResult(jobs=[], skipped=total_parts)
        # Only once there is real work to do, and before any of it is written. Asking
        # earlier would refuse a no-op batch for want of a host it was never going to use.
        self._require_capacity(execution)
        jobs: list[Job] = []
        for part_id in eligible:
            # One authorization and one model resolution per page, because job creation
            # lives in one place and this is the interface it offers. A chapter is a
            # handful of round trips per page against a warm session, and the alternative
            # is a second implementation of how a job row is built.
            job = await self._enqueue_one(
                session,
                user,
                project_id,
                document_id,
                part_id,
                job_type=job_type,
                execution=execution,
                model_id=model_id,
            )
            jobs.append(job)
        return BatchEnqueueResult(jobs=jobs, skipped=total_parts - len(jobs))

    async def _enqueue_one(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        job_type: JobType,
        execution: ExecutionRequest,
        model_id: UUID | None,
    ) -> Job:
        if job_type is JobType.segment:
            return await self._enqueue.enqueue_segment_part(
                session,
                user,
                project_id,
                document_id,
                part_id,
                execution=execution,
                model_id=model_id,
            )
        return await self._enqueue.enqueue_transcribe_part(
            session,
            user,
            project_id,
            document_id,
            part_id,
            execution=execution,
            model_id=model_id,
        )

    @staticmethod
    def _require_capacity(execution: ExecutionRequest) -> None:
        """Refuse the whole batch before the first job when no host can take work.

        Safe to decide once because it cannot change underneath us: the route reads
        capacity a single time and carries it down as a value (ADR 0002), so every page
        in this batch would reach the same verdict.

        A model that no *available* host may run is the one conflict this cannot see in
        advance, since it depends on the model resolved per page. That case still raises
        from inside the loop, and any jobs already committed stay queued - they are
        ordinary jobs on a host that can claim them, so cancelling them here would destroy
        work the researcher asked for in order to tidy up a count.
        """
        if not execution.available:
            raise ConflictError(NO_CAPACITY_MESSAGE)
