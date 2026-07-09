"""Persist browser-orchestrated local inference results into Postgres."""

from __future__ import annotations

import asyncio
import uuid
from uuid import UUID

from inference.contracts.segment import SegmentRunResponse
from inference.contracts.transcribe import TranscribeRunResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.application.document_service_shared import DocumentServiceSharedMixin
from backend.document.application.segment_merge_service import SegmentMergeService
from backend.document.application.transcribe_merge_service import (
    TranscribeJobHandlerError,
    TranscribeMergeService,
)
from backend.jobs.infrastructure.job_repository import JobRepository
from backend.jobs.infrastructure.orm_models import JobType
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import DocumentPart, Line
from backend.ml.infrastructure.ml_client import InferenceClient
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User
from infrastructure.db import sync_system_session


class LocalInferenceService(DocumentServiceSharedMixin):
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()

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
    ) -> UUID:
        job = await JobRepository(session).record_local_job(
            user_id=user.id,
            document_id=document_id,
            document_part_id=part_id,
            job_type=job_type,
            registry_model_id=registry_model_id,
            registry_tag=registry_tag,
            result=result,
        )
        return job.id

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
        project = await self._require_member(session, project_id, user.id)
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
        job_id = await self._record_local_job(
            session,
            user=user,
            document_id=document_id,
            part_id=part_id,
            job_type=JobType.transcribe,
            registry_model_id=registry_model_id,
            registry_tag=registry_tag,
            result=result,
        )
        result["job_id"] = str(job_id)
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
        project = await self._require_member(session, project_id, user.id)
        document = await self._load_document_in_project(session, project, document_id)
        await self._document_part_or_404(session, document, part_id)
        merge_job_id = uuid.uuid4()
        canonical = InferenceClient.to_canonical_segment(output)

        def _persist() -> dict:
            with sync_system_session() as sync_session:
                summary = SegmentMergeService().apply_sync(
                    sync_session,
                    part_id=part_id,
                    canonical_segment=canonical,
                    job_id=merge_job_id,
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

        result = await asyncio.to_thread(_persist)
        job_id = await self._record_local_job(
            session,
            user=user,
            document_id=document_id,
            part_id=part_id,
            job_type=JobType.segment,
            registry_model_id=registry_model_id,
            registry_tag=registry_tag,
            result=result,
        )
        result["job_id"] = str(job_id)
        return result
