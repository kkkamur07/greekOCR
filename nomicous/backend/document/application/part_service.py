"""Document parts: the page images, their order, their review state, and their bytes.

One responsibility — everything that treats a page as a *scan* rather than as geometry or
as text. That is what keeps the media store out of every other module in this context:
this is the only place that writes to it, reads from it, or has to compensate when a
write to it outlives the transaction that was supposed to record it.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError, ValidationError
from backend.document.application.document_access import DocumentAccess
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import (
    DEFAULT_PART_IMAGE_SUFFIX,
    MediaStore,
    encode_part_image_with_size,
    encode_part_thumbnail,
    get_media_store,
    read_image_size,
)
from backend.document.infrastructure.orm_models import DocumentPart
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User

logger = logging.getLogger(__name__)

# Parts uploaded before dimensions were persisted are backfilled lazily on read; bound how
# many object-store round trips a single request may trigger.
MAX_DIMENSION_BACKFILLS_PER_REQUEST = 25


class DocumentPartService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        media: MediaStore | None = None,
        access: DocumentAccess | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._media = media or get_media_store()
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)

    async def list_parts(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> list[DocumentPart]:
        context = await self._access.require_document(session, user, project_id, document_id)
        parts = sorted(context.document.parts, key=lambda p: p.order)
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
        context = await self._access.require_document(session, user, project_id, document_id)
        document = context.document
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
        context = await self._access.require_document(session, user, project_id, document_id)
        parts = await self._documents.reorder_parts(session, context.document, ordered_part_ids)
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
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        part = context.part
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
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        await self._documents.delete_part(session, context.part)

    async def get_part_for_media(
        self,
        session: AsyncSession,
        user: User,
        part_id: UUID,
    ) -> DocumentPart:
        context = await self._access.require_part_by_id(session, user, part_id)
        return context.part

    async def get_part_for_public_media(
        self,
        session: AsyncSession,
        part_id: UUID,
    ) -> DocumentPart:
        context = await self._access.require_part_by_id(session, None, part_id)
        return context.part

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
