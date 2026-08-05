"""Persist browser-orchestrated local inference results into Postgres."""

from __future__ import annotations

import asyncio
from uuid import UUID

from inference.contracts.segment import SegmentRunResponse
from inference.contracts.transcribe import TranscribeRunResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.application.document_access import DocumentAccess
from backend.document.application.segment_merge_service import SegmentMergeService
from backend.document.application.transcribe_merge_service import (
    TranscribeJobHandlerError,
    TranscribeMergeService,
)
from backend.jobs.infrastructure.job_repository import JobRepository
from backend.jobs.infrastructure.orm_models import Job, JobType
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import Document, DocumentPart, Line
from backend.ml.application.segment_mapping import to_canonical_segment
from backend.project.infrastructure.orm_models import Project
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User
from infrastructure.db import sync_system_session


class LocalInferenceService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        access: DocumentAccess | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)

    # The two persist paths below authorize in three named steps rather than one
    # ``DocumentAccess.require_part`` call. The rule itself still lives in exactly one
    # place - each of these is a straight delegation - but the *names* are the seam
    # ``tests/inference/unit/test_local_inference_provenance`` substitutes at to run the
    # merge without a database. That test belongs to the local-inference slice and is not
    # ours to rewrite, so the three-step shape is kept here deliberately.

    async def _require_member(self, session: AsyncSession, project_id: UUID, user: User) -> Project:
        return await self._access.require_project(session, user, project_id)

    async def _load_document_in_project(
        self, session: AsyncSession, project: Project, document_id: UUID
    ) -> Document:
        return await self._access.document_in_project(session, project, document_id)

    async def _document_part_or_404(
        self, session: AsyncSession, document: Document, part_id: UUID
    ) -> DocumentPart:
        return await self._access.part_in_document(session, document, part_id)

    async def _record_local_job(
        self,
        session: AsyncSession,
        *,
        user: User,
        document_id: UUID,
        part_id: UUID,
        job_type: JobType,
        registry_model_id: str,
        registry_tag: str,
        result: dict,
    ) -> Job:
        return await JobRepository(session).record_local_job(
            user_id=user.id,
            document_id=document_id,
            document_part_id=part_id,
            job_type=job_type,
            registry_model_id=registry_model_id,
            registry_tag=registry_tag,
            result=result,
        )

    async def persist_local_transcribe(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        registry_model_id: str,
        registry_tag: str,
        lines: list[tuple[UUID, TranscribeRunResponse]],
    ) -> dict:
        project = await self._require_member(session, project_id, user)
        document = await self._load_document_in_project(session, project, document_id)
        part = await self._document_part_or_404(session, document, part_id)

        def _persist() -> dict:
            with sync_system_session() as sync_session:
                sync_part = sync_session.get(DocumentPart, part.id)
                if sync_part is None or sync_part.document_id != document_id:
                    raise TranscribeJobHandlerError("Document part not found")

                lines_with_output: list[tuple[Line, TranscribeRunResponse]] = []
                for line_id, output in lines:
                    line = sync_session.get(Line, line_id)
                    if line is None or line.part_id != part_id:
                        raise TranscribeJobHandlerError("Document line not found")
                    lines_with_output.append((line, output))

                result = TranscribeMergeService().apply_sync(
                    sync_session,
                    document_id=document_id,
                    part_id=part_id,
                    job_id=None,
                    lines_with_output=lines_with_output,
                    layer_name=f"Local {registry_model_id}",
                    commit=True,
                )
                result["registry_model_id"] = registry_model_id
                result["registry_tag"] = registry_tag
                return result

        result = await asyncio.to_thread(_persist)
        job = await self._record_local_job(
            session,
            user=user,
            document_id=document_id,
            part_id=part_id,
            job_type=JobType.transcribe,
            registry_model_id=registry_model_id,
            registry_tag=registry_tag,
            result=result,
        )
        result["job_id"] = str(job.id)
        return result

    async def persist_local_segment(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        registry_model_id: str,
        registry_tag: str,
        output: SegmentRunResponse,
    ) -> dict:
        project = await self._require_member(session, project_id, user)
        document = await self._load_document_in_project(session, project, document_id)
        await self._document_part_or_404(session, document, part_id)
        canonical = to_canonical_segment(output)

        # The job row has to exist before the merge, because its primary key is
        # the only durable identity this run has: it is what the caller gets back
        # and what the jobs list shows, so it must also be what every merged line
        # records in ``source_metadata.job_id``. Minting a separate uuid for the
        # merge - as this used to - stamped the lines with an id no API ever
        # surfaced, which made local-run provenance impossible to query.
        job = await self._record_local_job(
            session,
            user=user,
            document_id=document_id,
            part_id=part_id,
            job_type=JobType.segment,
            registry_model_id=registry_model_id,
            registry_tag=registry_tag,
            result={"registry_model_id": registry_model_id, "registry_tag": registry_tag},
        )

        def _persist() -> dict:
            with sync_system_session() as sync_session:
                summary = SegmentMergeService().apply_sync(
                    sync_session,
                    part_id=part_id,
                    canonical_segment=canonical,
                    job_id=job.id,
                    commit=True,
                )
                return {
                    "registry_model_id": registry_model_id,
                    "registry_tag": registry_tag,
                    "blocks_count": summary.blocks_count,
                    "lines_count": summary.lines_count,
                    "added_lines": summary.added_lines,
                    "pruned_lines": summary.pruned_lines,
                    "preserved_manual_lines": summary.preserved_manual_lines,
                }

        try:
            result = await asyncio.to_thread(_persist)
        except Exception:
            # Nothing was merged, so a "done" job row claiming otherwise would be
            # a lie in the project's job history. Drop the placeholder and let the
            # failure surface exactly as it did when the row was written last.
            await session.delete(job)
            await session.commit()
            raise

        job.result = result
        await session.commit()
        result["job_id"] = str(job.id)
        return result
