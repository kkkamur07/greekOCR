"""Document part upload, ordering, review, and media access."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError, ValidationError
from backend.document.domain.access import require_can_read
from backend.document.infrastructure.media_store import (
    DEFAULT_PART_IMAGE_SUFFIX,
    encode_part_image_with_size,
    encode_part_thumbnail,
    read_image_size,
)
from backend.document.infrastructure.orm_models import DocumentPart
from backend.document.application.document_service_shared import DocumentServiceSharedMixin
from backend.users.infrastructure.orm_models import User

logger = logging.getLogger(__name__)

# Parts uploaded before dimensions were persisted are backfilled lazily on read; bound how
# many object-store round trips a single request may trigger.
MAX_DIMENSION_BACKFILLS_PER_REQUEST = 25


class PartServiceMixin(DocumentServiceSharedMixin):
    async def list_parts(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> list[DocumentPart]:
        document = await self.get_document(session, user, project_id, document_id)
        parts = sorted(document.parts, key=lambda p: p.order)
        await self.backfill_part_dimensions(session, parts)
        return parts

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
        # Single bounded decode: it validates the upload, produces the stored WebP, and
        # yields the dimensions the page canvas and PAGE XML export need.
        encoded = await asyncio.to_thread(encode_part_image_with_size, data)
        part = DocumentPart(
            document_id=document.id,
            order=order,
            image_key="pending",
            width=encoded.width,
            height=encoded.height,
        )
        session.add(part)
        await session.flush()
        image_key = self._media.part_image_key(
            part.id,
            suffix=DEFAULT_PART_IMAGE_SUFFIX,
            filename_stem=filename_stem,
        )
        try:
            self._media.write(image_key, encoded.data)
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

    async def backfill_part_dimensions(
        self,
        session: AsyncSession,
        parts: list[DocumentPart],
    ) -> None:
        """Fill width/height for parts stored before dimensions were persisted.

        Dimensions only exist inside the stored blob, so migration 004 cannot backfill
        them in SQL. Read paths that expose dimensions decode the stored image once and
        persist the result; every later read is served from Postgres. Storage failures are
        swallowed — a missing blob must not break listing the rest of the document.
        """
        pending = [part for part in parts if part.width is None or part.height is None]
        if not pending:
            return
        changed = False
        for part in pending[:MAX_DIMENSION_BACKFILLS_PER_REQUEST]:
            try:
                size = await asyncio.to_thread(self._read_part_image_size, part)
            except Exception:
                logger.warning("Could not recover dimensions for part %s", part.id, exc_info=True)
                continue
            part.width, part.height = size
            changed = True
        if changed:
            await session.commit()

    def _read_part_image_size(self, part: DocumentPart) -> tuple[int, int]:
        return read_image_size(self._media.read(part.image_key))

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
