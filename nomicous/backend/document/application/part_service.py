"""Document part upload, ordering, review, and media access."""

import asyncio
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError, ValidationError
from backend.document.domain.access import require_can_read
from backend.document.infrastructure.media_store import (
    DEFAULT_PART_IMAGE_SUFFIX,
    encode_part_image,
    encode_part_thumbnail,
)
from backend.document.infrastructure.orm_models import DocumentPart
from backend.document.application.document_service_shared import DocumentServiceSharedMixin
from backend.users.infrastructure.orm_models import User

logger = logging.getLogger(__name__)


class PartServiceMixin(DocumentServiceSharedMixin):
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
        filename_stem: str | None = None
        if filename and "." in filename:
            filename_stem = filename.rsplit(".", 1)[0]
        encoded = encode_part_image(data)
        part = DocumentPart(document_id=document.id, order=order, image_key="pending")
        session.add(part)
        await session.flush()
        image_key = self._media.part_image_key(
            part.id,
            suffix=DEFAULT_PART_IMAGE_SUFFIX,
            filename_stem=filename_stem,
        )
        try:
            self._media.write(image_key, encoded)
            part.image_key = image_key
            await session.commit()
        except Exception:
            await session.rollback()
            try:
                self._media.delete(image_key)
            except Exception:
                try:
                    await self._documents.enqueue_media_deletion_intent(session, image_key)
                except Exception:
                    await session.rollback()
                    logger.exception("Could not persist media compensation intent")
            raise
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

    async def update_part_review_status(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        reviewed: bool,
    ) -> DocumentPart:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        part.reviewed = reviewed
        await session.commit()
        await session.refresh(part)
        return part

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
        await self._documents.delete_part(session, part)

    async def get_part_for_media(
        self,
        session: AsyncSession,
        user: User,
        part_id: UUID,
    ) -> DocumentPart:
        part = await self._documents.get_part_row(session, part_id)
        if part is None:
            raise NotFoundError("Part not found")
        document = await self._documents.get_by_id_for_authz(session, part.document_id)
        if document is None:
            raise NotFoundError("Document not found")
        await self._require_member(session, document.project_id, user.id)
        return part

    async def get_part_for_public_media(
        self,
        session: AsyncSession,
        part_id: UUID,
    ) -> DocumentPart:
        part = await self._documents.get_part_row(session, part_id)
        if part is None:
            raise NotFoundError("Part not found")
        document = await self._documents.get_by_id_for_authz(session, part.document_id)
        if document is None:
            raise NotFoundError("Document not found")
        project = await self._load_project(session, document.project_id)
        require_can_read(document, project, None)
        return part

    async def read_part_bytes(self, part: DocumentPart, *, width: int | None = None) -> bytes:
        """Read and optionally transform media without blocking the event loop."""
        return await asyncio.to_thread(self._read_part_bytes, part, width)

    def _read_part_bytes(self, part: DocumentPart, width: int | None) -> bytes:
        try:
            data = self._media.read(part.image_key)
        except (ValueError, FileNotFoundError):
            raise NotFoundError("Part image not found") from None
        if width is None:
            return data
        return encode_part_thumbnail(data, width)
