"""Enqueue ML jobs for document parts."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.document.application.document_service_shared import DocumentServiceSharedMixin
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.ml.infrastructure.orm_models import InferenceTask
from backend.users.infrastructure.orm_models import User


class DocumentJobEnqueueMixin(DocumentServiceSharedMixin):
    async def enqueue_transcribe_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        model_id: UUID | None = None,
        line_ids: list[UUID] | None = None,
    ) -> Job:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        lines = await self._documents.list_part_lines(session, part.id)
        if not lines:
            raise ConflictError("Cannot transcribe a document part without layout lines")
        if line_ids is not None:
            selected_ids = set(line_ids)
            known_ids = {line.id for line in lines}
            if selected_ids - known_ids:
                raise NotFoundError("Line not found")
            if not selected_ids:
                raise ValidationError("At least one line must be selected for transcription")
        binding_id: UUID | None = None
        selected_model_id = model_id
        ml_params: dict = {}
        if selected_model_id is not None:
            model = await self._inference_models.get_model_for_task(
                session, selected_model_id, InferenceTask.transcribe
            )
            ml_params = dict(model.default_params or {})
        else:
            try:
                resolved = await self._inference_models.resolve_for_part(
                    session,
                    user,
                    project_id,
                    document_id,
                    part_id,
                    task=InferenceTask.transcribe,
                )
            except NotFoundError:
                selected_model_id = None
            else:
                selected_model_id = resolved.model.id
                binding_id = resolved.binding.id
                ml_params = dict(resolved.effective_params)
        payload: dict = {"ml_params": ml_params, "execution": "cloud"}
        if line_ids is not None:
            payload["line_ids"] = [str(line_id) for line_id in line_ids]
        job = Job(
            type=JobType.transcribe,
            status=JobStatus.pending,
            user_id=user.id,
            document_id=document.id,
            document_part_id=part.id,
            model_id=selected_model_id,
            binding_id=binding_id,
            payload=payload,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job

    async def enqueue_segment_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        model_id: UUID | None = None,
        ml_params: dict | None = None,
    ) -> Job:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        binding_id: UUID | None = None
        selected_model_id = model_id
        effective_params: dict = dict(ml_params or {})
        if selected_model_id is not None:
            model = await self._inference_models.get_model_for_task(
                session, selected_model_id, InferenceTask.segment
            )
            resolved_params = dict(model.default_params or {})
            resolved_params.update(effective_params)
            effective_params = resolved_params
        else:
            try:
                resolved = await self._inference_models.resolve_for_part(
                    session,
                    user,
                    project_id,
                    document_id,
                    part_id,
                    task=InferenceTask.segment,
                )
            except NotFoundError:
                selected_model_id = None
            else:
                selected_model_id = resolved.model.id
                binding_id = resolved.binding.id
                merged_params = dict(resolved.effective_params)
                merged_params.update(effective_params)
                effective_params = merged_params
        job = Job(
            type=JobType.segment,
            status=JobStatus.pending,
            user_id=user.id,
            document_id=document.id,
            document_part_id=part.id,
            model_id=selected_model_id,
            binding_id=binding_id,
            payload={"ml_params": effective_params, "execution": "cloud"},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job
