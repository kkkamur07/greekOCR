"""Document CRUD and public read use cases."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ValidationError
from backend.document.domain.access import require_can_read
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    DocumentWorkflow,
    Line,
    Transcription,
)
from backend.document.application.document_service_shared import (
    DOCUMENT_UPDATE_FIELDS,
    DocumentServiceSharedMixin,
)
from backend.users.infrastructure.orm_models import User


class DocumentCrudMixin(DocumentServiceSharedMixin):
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

    async def get_published_part(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> DocumentPart:
        document = await self.get_document_public(session, project_id, document_id)
        return await self._document_part_or_404(session, document, part_id)

    async def update_document(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        **fields: object,
    ) -> Document:
        self._reject_unknown_fields(fields, DOCUMENT_UPDATE_FIELDS, "document update")
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
        image_keys = [part.image_key for part in document.parts]
        for image_key in image_keys:
            self._media.delete(image_key)
        await self._documents.delete(session, document)

    async def list_transcriptions(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> list[Transcription]:
        document = await self.get_document(session, user, project_id, document_id)
        return await self._documents.list_transcriptions(session, document.id)

    async def list_transcriptions_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
    ) -> list[Transcription]:
        document = await self.get_document_public(session, project_id, document_id)
        return await self._documents.list_transcriptions(session, document.id)

    async def list_document_layout_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
    ) -> tuple[list[Block], list[Line]]:
        document = await self.get_document_public(session, project_id, document_id)
        blocks: list[Block] = []
        lines: list[Line] = []
        for part in sorted(document.parts, key=lambda item: item.order):
            blocks.extend(await self._list_part_blocks(session, part.id))
            lines.extend(await self._documents.list_part_lines(session, part.id))
        return blocks, lines
