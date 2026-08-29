"""Set-shaped reads over a whole document: which pages still need work, and how many.

A per-page route can afford to load a part with its lines and count in Python. A chapter
cannot: the numbers behind menu labels like "Segment unsegmented pages 12" are recomputed
every time that menu opens, and the fan-out routes need the eligible set *before* they
write anything. So every query here answers a question about all the parts of one
document in a single statement, and returns ids rather than rows.

Two definitions live here rather than in the service, because they are what the SQL
means:

* a part is **unsegmented** when it has no ``lines`` row. Segmentation is the only thing
  that creates those rows, so their absence is the durable evidence that this page has
  never been segmented, and it survives a restart in a way an in-memory flag would not.
* a part is **unpaired** when it has lines but not one of them carries transcription
  text. ``kind`` is deliberately not filtered: a page a researcher already typed out by
  hand needs the model run over it no more than one a model has already transcribed.
  Blank text does not count, the same rule ``count_paired_ground_truth_lines`` applies
  next door, because a layer of empty strings is not a transcription anyone can read.

A part with no lines is therefore never *unpaired*: transcription runs on segments, and
the per-part service refuses a page without them. Counting such a page in the "transcribe
unpaired pages" label would promise work the platform would then decline to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from backend.document.infrastructure.orm_models import DocumentPart, Line, LineTranscription
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType

#: The statuses that mean a job for this page is still going to run. A second job for the
#: same page and task would double the inference spend and race the first one's write, so
#: a part in any of these is skipped rather than queued again. ``done``/``failed``/
#: ``cancelled`` are absent on purpose: re-running a page whose job finished, or died, is
#: exactly what a batch action is for.
IN_FLIGHT_JOB_STATUSES = (JobStatus.pending, JobStatus.waiting, JobStatus.running)


@dataclass(frozen=True)
class WorkflowCounts:
    """How far a document has got, in the four numbers the batch menu renders."""

    total: int
    reviewed: int
    unsegmented: int
    unpaired: int


def _has_lines() -> ColumnElement[bool]:
    """Correlated EXISTS: this part has been segmented at least once."""
    return select(1).select_from(Line).where(Line.part_id == DocumentPart.id).exists()


def _has_transcription_text() -> ColumnElement[bool]:
    """Correlated EXISTS: some segment on this part carries non-blank text."""
    return (
        select(1)
        .select_from(Line)
        .join(LineTranscription, LineTranscription.line_id == Line.id)
        .where(
            Line.part_id == DocumentPart.id,
            func.length(func.trim(LineTranscription.text)) > 0,
        )
        .exists()
    )


class DocumentBatchRepository:
    async def workflow_counts(self, session: AsyncSession, document_id: UUID) -> WorkflowCounts:
        """All four counts in one statement, computed by Postgres.

        Aggregate ``FILTER`` clauses rather than four round trips, and rather than
        loading the parts: the caller renders these into labels, so they have to be
        exact, and a chapter is hundreds of pages with thousands of lines under it.
        """
        has_lines = _has_lines()
        has_text = _has_transcription_text()
        result = await session.execute(
            select(
                func.count(),
                func.count().filter(DocumentPart.reviewed.is_(True)),
                func.count().filter(~has_lines),
                func.count().filter(has_lines, ~has_text),
            ).where(DocumentPart.document_id == document_id)
        )
        total, reviewed, unsegmented, unpaired = result.one()
        return WorkflowCounts(
            total=int(total),
            reviewed=int(reviewed),
            unsegmented=int(unsegmented),
            unpaired=int(unpaired),
        )

    async def part_ids_to_segment(
        self, session: AsyncSession, document_id: UUID, *, only_unsegmented: bool
    ) -> list[UUID]:
        """Candidate pages for segmentation, in reading order.

        Reading order is not cosmetic: the jobs are claimed roughly in the order they are
        written, so a researcher watching a chapter go through sees page 1 come back
        first rather than whichever page Postgres happened to return first.
        """
        stmt = select(DocumentPart.id).where(DocumentPart.document_id == document_id)
        if only_unsegmented:
            stmt = stmt.where(~_has_lines())
        return await self._part_ids(session, stmt)

    async def part_ids_to_transcribe(
        self, session: AsyncSession, document_id: UUID, *, only_unpaired: bool
    ) -> list[UUID]:
        """Candidate pages for transcription, in reading order.

        ``_has_lines()`` is applied under *both* scopes, not just the narrow one: even
        "transcribe every page" cannot mean a page with no segments, because there would
        be nothing to run the model over.
        """
        stmt = select(DocumentPart.id).where(DocumentPart.document_id == document_id, _has_lines())
        if only_unpaired:
            stmt = stmt.where(~_has_transcription_text())
        return await self._part_ids(session, stmt)

    async def part_ids_with_job_in_flight(
        self, session: AsyncSession, document_id: UUID, *, job_type: JobType
    ) -> set[UUID]:
        """Parts of this document already carrying an unfinished job of this type."""
        result = await session.execute(
            select(Job.document_part_id)
            .where(
                Job.document_id == document_id,
                Job.type == job_type,
                Job.status.in_(IN_FLIGHT_JOB_STATUSES),
                Job.document_part_id.is_not(None),
            )
            .distinct()
        )
        return {part_id for part_id in result.scalars() if part_id is not None}

    async def _part_ids(self, session: AsyncSession, stmt: Select[tuple[UUID]]) -> list[UUID]:
        result = await session.execute(stmt.order_by(DocumentPart.order, DocumentPart.id))
        return list(result.scalars().all())
