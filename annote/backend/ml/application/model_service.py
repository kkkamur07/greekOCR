"""ML catalog, scoped bindings, and resolver use cases."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import AccessDeniedError, ConflictError, NotFoundError, ValidationError
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import Document, DocumentPart
from backend.ml.infrastructure.model_repository import MlRepository
from backend.ml.infrastructure.orm_models import (
    InferenceModel,
    InferenceTask,
    ModelBinding,
)
from backend.project.domain.access import is_member
from backend.project.infrastructure.orm_models import Project
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User


@dataclass(frozen=True)
class ResolvedModelBinding:
    binding: ModelBinding
    model: InferenceModel
    effective_params: dict


class InferenceModelService:
    def __init__(
        self,
        inference: MlRepository | None = None,
        projects: ProjectRepository | None = None,
        documents: DocumentRepository | None = None,
    ) -> None:
        self._inference = inference or MlRepository()
        self._projects = projects or ProjectRepository()
        self._documents = documents or DocumentRepository()

    async def list_models(self, session: AsyncSession) -> list[InferenceModel]:
        return await self._inference.list_models(session)

    async def get_model_for_task(
        self, session: AsyncSession, model_id: UUID, task: InferenceTask
    ) -> InferenceModel:
        return await self._require_model_for_task(session, model_id, task)

    async def list_project_bindings(
        self, session: AsyncSession, user: User, project_id: UUID
    ) -> list[ModelBinding]:
        await self._require_project_member(session, user, project_id)
        return await self._inference.list_bindings(session, project_id=project_id)

    async def create_project_binding(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        *,
        task: InferenceTask,
        model_id: UUID,
        overrides: dict | None = None,
    ) -> ModelBinding:
        await self._require_project_member(session, user, project_id)
        await self._require_model_for_task(session, model_id, task)
        await self._require_no_existing_binding(session, task=task, project_id=project_id)
        return await self._inference.create_binding(
            session,
            task=task,
            model_id=model_id,
            project_id=project_id,
            overrides=overrides,
        )

    async def list_document_bindings(
        self, session: AsyncSession, user: User, project_id: UUID, document_id: UUID
    ) -> list[ModelBinding]:
        await self._require_document_in_project(session, user, project_id, document_id)
        return await self._inference.list_bindings(session, document_id=document_id)

    async def create_document_binding(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        task: InferenceTask,
        model_id: UUID,
        overrides: dict | None = None,
    ) -> ModelBinding:
        await self._require_document_in_project(session, user, project_id, document_id)
        await self._require_model_for_task(session, model_id, task)
        await self._require_no_existing_binding(session, task=task, document_id=document_id)
        return await self._inference.create_binding(
            session,
            task=task,
            model_id=model_id,
            document_id=document_id,
            overrides=overrides,
        )

    async def list_part_bindings(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> list[ModelBinding]:
        await self._require_part_in_document(session, user, project_id, document_id, part_id)
        return await self._inference.list_bindings(session, document_part_id=part_id)

    async def create_part_binding(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        task: InferenceTask,
        model_id: UUID,
        overrides: dict | None = None,
    ) -> ModelBinding:
        await self._require_part_in_document(session, user, project_id, document_id, part_id)
        await self._require_model_for_task(session, model_id, task)
        await self._require_no_existing_binding(session, task=task, document_part_id=part_id)
        return await self._inference.create_binding(
            session,
            task=task,
            model_id=model_id,
            document_part_id=part_id,
            overrides=overrides,
        )

    async def update_binding(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        binding_id: UUID,
        *,
        model_id: UUID | None = None,
        overrides: dict | None = None,
    ) -> ModelBinding:
        binding = await self._binding_visible_to_project(session, user, project_id, binding_id)
        if model_id is not None:
            await self._require_model_for_task(session, model_id, binding.task)
        return await self._inference.update_binding(
            session, binding, model_id=model_id, overrides=overrides
        )

    async def delete_binding(
        self, session: AsyncSession, user: User, project_id: UUID, binding_id: UUID
    ) -> None:
        binding = await self._binding_visible_to_project(session, user, project_id, binding_id)
        await self._inference.delete_binding(session, binding)

    async def resolve_for_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        task: InferenceTask,
    ) -> ResolvedModelBinding:
        await self._require_part_in_document(session, user, project_id, document_id, part_id)
        for scope in (
            {"document_part_id": part_id},
            {"document_id": document_id},
            {"project_id": project_id},
        ):
            binding = await self._inference.find_binding(session, task=task, **scope)
            if binding is not None:
                params = dict(binding.model.default_params or {})
                params.update(binding.overrides or {})
                return ResolvedModelBinding(
                    binding=binding,
                    model=binding.model,
                    effective_params=params,
                )
        raise NotFoundError(f"No {task.value} model binding found")

    async def _require_model_for_task(
        self, session: AsyncSession, model_id: UUID, task: InferenceTask
    ) -> InferenceModel:
        model = await self._inference.get_model(session, model_id)
        if model is None:
            raise NotFoundError("Inference model not found")
        if model.task != task:
            raise ValidationError("Model task does not match binding task")
        return model

    async def _require_no_existing_binding(
        self,
        session: AsyncSession,
        *,
        task: InferenceTask,
        project_id: UUID | None = None,
        document_id: UUID | None = None,
        document_part_id: UUID | None = None,
    ) -> None:
        existing = await self._inference.find_binding(
            session,
            task=task,
            project_id=project_id,
            document_id=document_id,
            document_part_id=document_part_id,
        )
        if existing is not None:
            raise ConflictError("A binding already exists for this scope and task")

    async def _require_project_member(
        self, session: AsyncSession, user: User, project_id: UUID
    ) -> Project:
        project = await self._projects.get_by_id(session, project_id)
        if project is None:
            raise NotFoundError("Project not found")
        if not is_member(project, user.id):
            raise AccessDeniedError("You do not have access to this project")
        return project

    async def _require_document_in_project(
        self, session: AsyncSession, user: User, project_id: UUID, document_id: UUID
    ) -> Document:
        await self._require_project_member(session, user, project_id)
        document = await self._documents.get_by_id(session, document_id)
        if document is None or document.project_id != project_id:
            raise NotFoundError("Document not found")
        return document

    async def _require_part_in_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> DocumentPart:
        await self._require_document_in_project(session, user, project_id, document_id)
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document_id:
            raise NotFoundError("Document part not found")
        return part

    async def _binding_visible_to_project(
        self, session: AsyncSession, user: User, project_id: UUID, binding_id: UUID
    ) -> ModelBinding:
        await self._require_project_member(session, user, project_id)
        binding = await self._inference.get_binding(session, binding_id)
        if binding is None:
            raise NotFoundError("Model binding not found")
        if binding.project_id == project_id:
            return binding
        if binding.document_id is not None:
            document = await self._documents.get_by_id(session, binding.document_id)
            if document is not None and document.project_id == project_id:
                return binding
        if binding.document_part_id is not None:
            part = await self._documents.get_part(session, binding.document_part_id)
            if part is not None:
                document = await self._documents.get_by_id(session, part.document_id)
                if document is not None and document.project_id == project_id:
                    return binding
        raise NotFoundError("Model binding not found")
