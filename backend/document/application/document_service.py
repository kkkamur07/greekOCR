"""Document and part use cases with project membership checks."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import AccessDeniedError, NotFoundError, ValidationError
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import MediaStore
from backend.document.infrastructure.orm_models import Document, DocumentPart, DocumentWorkflow
from backend.document.domain.access import require_can_read
from backend.project.domain.access import is_member
from backend.project.infrastructure.orm_models import Project
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User


class DocumentService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        media: MediaStore | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._media = media or MediaStore()

    async def list_documents(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        *,
        include_archived: bool = False,
    ) -> list[Document]:
        await self._require_member(session, project_id, user.id)
        return await self._documents.list_for_project(
            session, project_id, include_archived=include_archived
        )

    async def create_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        *,
        name: str,
    ) -> Document:
        await self._require_member(session, project_id, user.id)
        return await self._documents.create(session, project_id=project_id, name=name)

    async def get_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> Document:
        project = await self._require_member(session, project_id, user.id)
        document = await self._load_document_in_project(session, project, document_id)
        return document

    async def get_document_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
    ) -> Document:
        project = await self._load_project(session, project_id)
        document = await self._load_document_in_project(session, project, document_id)
        require_can_read(document, project, None)
        return document

    async def update_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        **fields: object,
    ) -> Document:
        project = await self._require_member(session, project_id, user.id)
        document = await self._load_document_in_project(session, project, document_id)
        if "workflow" in fields and fields["workflow"] is not None:
            workflow = fields["workflow"]
            if not isinstance(workflow, DocumentWorkflow):
                raise ValidationError("Invalid workflow value")
        return await self._documents.update(session, document, **fields)

    async def delete_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> None:
        project = await self._require_member(session, project_id, user.id)
        document = await self._load_document_in_project(session, project, document_id)
        for part in list(document.parts):
            self._media.delete(part.image_key)
        await self._documents.delete(session, document)

    async def list_parts(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> list[DocumentPart]:
        document = await self.get_document(session, user, project_id, document_id)
        return sorted(document.parts, key=lambda p: p.order)

    async def upload_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        data: bytes,
        filename: str | None = None,
    ) -> DocumentPart:
        document = await self.get_document(session, user, project_id, document_id)
        order = await self._documents.next_part_order(session, document.id)
        suffix = "bin"
        if filename and "." in filename:
            suffix = filename.rsplit(".", 1)[-1].lower()[:16]
        part = DocumentPart(document_id=document.id, order=order, image_key="pending")
        session.add(part)
        await session.flush()
        image_key = self._media.part_image_key(part.id, suffix=suffix)
        self._media.write(image_key, data)
        part.image_key = image_key
        await session.commit()
        await session.refresh(part)
        return part

    async def reorder_parts(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        ordered_part_ids: list[UUID],
    ) -> list[DocumentPart]:
        document = await self.get_document(session, user, project_id, document_id)
        parts = await self._documents.reorder_parts(session, document, ordered_part_ids)
        if not parts:
            raise ValidationError("part_ids must match all parts on the document")
        return parts

    async def delete_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> None:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document.id:
            raise NotFoundError("Part not found")
        self._media.delete(part.image_key)
        await self._documents.delete_part(session, part)

    async def get_part_for_media(
        self,
        session: AsyncSession,
        user: User,
        part_id: UUID,
    ) -> DocumentPart:
        part = await self._documents.get_part(session, part_id)
        if part is None:
            raise NotFoundError("Part not found")
        document = await self._documents.get_by_id(session, part.document_id)
        if document is None:
            raise NotFoundError("Document not found")
        await self._require_member(session, document.project_id, user.id)
        return part

    async def get_part_for_public_media(
        self,
        session: AsyncSession,
        part_id: UUID,
    ) -> DocumentPart:
        part = await self._documents.get_part(session, part_id)
        if part is None:
            raise NotFoundError("Part not found")
        document = await self._documents.get_by_id(session, part.document_id)
        if document is None:
            raise NotFoundError("Document not found")
        project = await self._load_project(session, document.project_id)
        require_can_read(document, project, None)
        return part

    def read_part_bytes(self, part: DocumentPart) -> bytes:
        return self._media.read(part.image_key)

    async def _load_project(self, session: AsyncSession, project_id: UUID) -> Project:
        project = await self._projects.get_by_id(session, project_id)
        if project is None:
            raise NotFoundError("Project not found")
        return project

    async def _require_member(
        self, session: AsyncSession, project_id: UUID, user_id: UUID
    ) -> Project:
        project = await self._load_project(session, project_id)
        if not is_member(project, user_id):
            raise AccessDeniedError("You do not have access to this project")
        return project

    async def _load_document_in_project(
        self, session: AsyncSession, project: Project, document_id: UUID
    ) -> Document:
        document = await self._documents.get_by_id(session, document_id)
        if document is None or document.project_id != project.id:
            raise NotFoundError("Document not found")
        return document
