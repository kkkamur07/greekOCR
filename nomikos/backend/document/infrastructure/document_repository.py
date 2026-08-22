"""Document and DocumentPart persistence."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.core.api.pagination import PageCursor
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    DocumentWorkflow,
    Line,
    LineTranscription,
    MediaDeletionIntent,
    PageTranscriptionLine,
    Transcription,
    TranscriptionKind,
)


def temporary_reorder_offset(current_orders: Sequence[int], target_count: int) -> int:
    """Offset that parks every row below the ``[0, target_count)`` range being written.

    ``uq_document_parts_document_order`` is checked per statement, so the shift has to
    clear both the orders still in the table and the orders about to be assigned.
    Anchoring at ``min(minimum_order, 0)`` matters: when the surviving parts no longer
    start at 0 (the first pages were deleted), anchoring at ``minimum_order`` leaves the
    shifted rows inside the target range and the reorder collides with itself.
    """
    minimum_order = min(current_orders)
    maximum_order = max(current_orders)
    return min(minimum_order, 0) - maximum_order - target_count - 1


class DocumentRepository:
    async def get_by_id(self, session: AsyncSession, document_id: UUID) -> Document | None:
        result = await session.execute(
            select(Document)
            .options(selectinload(Document.parts))
            .options(selectinload(Document.transcriptions))
            .where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id_for_authz(
        self, session: AsyncSession, document_id: UUID
    ) -> Document | None:
        """Load a document row without parts/transcriptions (media authz hot path)."""
        result = await session.execute(select(Document).where(Document.id == document_id))
        return result.scalar_one_or_none()

    async def list_for_project(
        self,
        session: AsyncSession,
        project_id: UUID,
        *,
        include_archived: bool = False,
        limit: int = 50,
        cursor: PageCursor | None = None,
    ) -> list[Document]:
        stmt = select(Document).where(Document.project_id == project_id)
        if not include_archived:
            stmt = stmt.where(Document.workflow != DocumentWorkflow.archived)
        stmt = stmt.order_by(Document.created_at.desc(), Document.id.desc())
        if cursor is not None:
            stmt = stmt.where(
                tuple_(Document.created_at, Document.id) < (cursor.created_at, cursor.id)
            )
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def count_parts_by_document_ids(
        self, session: AsyncSession, document_ids: list[UUID]
    ) -> dict[UUID, int]:
        if not document_ids:
            return {}
        result = await session.execute(
            select(DocumentPart.document_id, func.count())
            .where(DocumentPart.document_id.in_(document_ids))
            .group_by(DocumentPart.document_id)
        )
        return {document_id: int(count) for document_id, count in result.all()}

    async def count_documents_by_project_ids(
        self, session: AsyncSession, project_ids: list[UUID]
    ) -> dict[UUID, int]:
        if not project_ids:
            return {}
        result = await session.execute(
            select(Document.project_id, func.count())
            .where(Document.project_id.in_(project_ids))
            .group_by(Document.project_id)
        )
        return {project_id: int(count) for project_id, count in result.all()}

    async def create(
        self,
        session: AsyncSession,
        *,
        project_id: UUID,
        name: str,
        workflow: DocumentWorkflow = DocumentWorkflow.draft,
    ) -> Document:
        document = Document(project_id=project_id, name=name, workflow=workflow)
        document.transcriptions.append(
            Transcription(
                name="Ground truth",
                kind=TranscriptionKind.ground_truth,
            )
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)
        return document

    async def update(
        self,
        session: AsyncSession,
        document: Document,
        **fields: object,
    ) -> Document:
        for key, value in fields.items():
            setattr(document, key, value)
        await session.commit()
        await session.refresh(document)
        return document

    async def delete(self, session: AsyncSession, document: Document) -> None:
        await session.delete(document)
        await session.commit()

    async def delete_with_media_intents(
        self,
        session: AsyncSession,
        document: Document,
        image_keys: list[str],
    ) -> None:
        for image_key in image_keys:
            session.add(MediaDeletionIntent(image_key=image_key))
        await session.delete(document)
        await session.commit()

    async def get_part(self, session: AsyncSession, part_id: UUID) -> DocumentPart | None:
        result = await session.execute(
            select(DocumentPart)
            .options(
                selectinload(DocumentPart.lines)
                .selectinload(Line.transcriptions)
                .selectinload(LineTranscription.transcription)
            )
            .where(DocumentPart.id == part_id)
        )
        return result.scalar_one_or_none()

    async def get_part_row(self, session: AsyncSession, part_id: UUID) -> DocumentPart | None:
        """Load a part without lines/transcriptions (media serving authz)."""
        result = await session.execute(select(DocumentPart).where(DocumentPart.id == part_id))
        return result.scalar_one_or_none()

    async def list_transcriptions(
        self, session: AsyncSession, document_id: UUID
    ) -> list[Transcription]:
        result = await session.execute(
            select(Transcription)
            .where(Transcription.document_id == document_id)
            .order_by(Transcription.created_at, Transcription.id)
        )
        return list(result.scalars().all())

    async def get_ground_truth_transcription(
        self, session: AsyncSession, document_id: UUID
    ) -> Transcription | None:
        result = await session.execute(
            select(Transcription).where(
                Transcription.document_id == document_id,
                Transcription.kind == TranscriptionKind.ground_truth,
            )
        )
        return result.scalar_one_or_none()

    async def list_part_lines(self, session: AsyncSession, part_id: UUID) -> list[Line]:
        result = await session.execute(
            select(Line)
            .options(
                selectinload(Line.transcriptions).selectinload(LineTranscription.transcription)
            )
            .where(Line.part_id == part_id)
            .order_by(Line.order, Line.created_at)
        )
        return list(result.scalars().all())

    async def list_blocks_for_document(
        self, session: AsyncSession, document_id: UUID, *, limit: int
    ) -> list[Block]:
        result = await session.execute(
            select(Block)
            .join(DocumentPart, Block.part_id == DocumentPart.id)
            .where(DocumentPart.document_id == document_id)
            .order_by(DocumentPart.order, Block.order, Block.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_lines_for_document(
        self,
        session: AsyncSession,
        document_id: UUID,
        *,
        limit: int,
        cursor: PageCursor | None = None,
    ) -> list[Line]:
        """Keyset page over a document's lines.

        Ordered by ``(created_at, id)`` so the shared ``PageCursor`` applies; clients
        group by part and sort by ``order`` themselves.
        """
        stmt = (
            select(Line)
            .options(
                selectinload(Line.transcriptions).selectinload(LineTranscription.transcription)
            )
            .join(DocumentPart, Line.part_id == DocumentPart.id)
            .where(DocumentPart.document_id == document_id)
            .order_by(Line.created_at, Line.id)
        )
        if cursor is not None:
            stmt = stmt.where(tuple_(Line.created_at, Line.id) > (cursor.created_at, cursor.id))
        result = await session.execute(stmt.limit(limit))
        return list(result.scalars().all())

    async def list_part_blocks(self, session: AsyncSession, part_id: UUID) -> list[Block]:
        result = await session.execute(
            select(Block).where(Block.part_id == part_id).order_by(Block.order, Block.created_at)
        )
        return list(result.scalars().all())

    async def get_block_in_part(
        self, session: AsyncSession, part_id: UUID, block_id: UUID
    ) -> Block | None:
        result = await session.execute(
            select(Block).where(Block.id == block_id, Block.part_id == part_id)
        )
        return result.scalar_one_or_none()

    async def get_line_in_part(
        self, session: AsyncSession, part_id: UUID, line_id: UUID
    ) -> Line | None:
        result = await session.execute(
            select(Line)
            .where(Line.id == line_id, Line.part_id == part_id)
            .options(
                selectinload(Line.transcriptions).selectinload(LineTranscription.transcription)
            )
        )
        return result.scalar_one_or_none()

    async def get_line_in_document(
        self, session: AsyncSession, document_id: UUID, line_id: UUID
    ) -> Line | None:
        result = await session.execute(
            select(Line)
            .join(DocumentPart, Line.part_id == DocumentPart.id)
            .where(Line.id == line_id, DocumentPart.document_id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_transcription_in_document(
        self, session: AsyncSession, document_id: UUID, transcription_id: UUID
    ) -> Transcription | None:
        transcription = await session.get(Transcription, transcription_id)
        if transcription is None or transcription.document_id != document_id:
            return None
        return transcription

    async def count_part_lines(self, session: AsyncSession, part_id: UUID) -> int:
        result = await session.execute(
            select(func.count()).select_from(Line).where(Line.part_id == part_id)
        )
        return int(result.scalar_one())

    async def count_paired_ground_truth_lines(self, session: AsyncSession, part_id: UUID) -> int:
        """Lines carrying non-blank ground-truth text — the denominator of pairing progress."""
        result = await session.execute(
            select(func.count(func.distinct(Line.id)))
            .select_from(Line)
            .join(LineTranscription, LineTranscription.line_id == Line.id)
            .join(Transcription, LineTranscription.transcription_id == Transcription.id)
            .where(
                Line.part_id == part_id,
                Transcription.kind == TranscriptionKind.ground_truth,
                func.length(func.trim(LineTranscription.text)) > 0,
            )
        )
        return int(result.scalar_one())

    async def list_page_transcription_lines(
        self, session: AsyncSession, part_id: UUID
    ) -> list[PageTranscriptionLine]:
        result = await session.execute(
            select(PageTranscriptionLine)
            .where(PageTranscriptionLine.part_id == part_id)
            .order_by(PageTranscriptionLine.order, PageTranscriptionLine.created_at)
        )
        return list(result.scalars().all())

    async def get_page_transcription_line(
        self, session: AsyncSession, part_id: UUID, order: int
    ) -> PageTranscriptionLine | None:
        result = await session.execute(
            select(PageTranscriptionLine).where(
                PageTranscriptionLine.part_id == part_id,
                PageTranscriptionLine.order == order,
            )
        )
        return result.scalar_one_or_none()

    async def part_page_number(self, session: AsyncSession, part: DocumentPart) -> int:
        """1-based position of ``part`` among its document's parts, in display order.

        Computed from the rows rather than read off ``part.order`` because orders are
        allowed to have gaps: ``next_part_order`` hands out max+1, so deleting a middle
        part leaves a hole, and the UI numbers pages by position, not by order value.
        """
        result = await session.execute(
            select(func.count())
            .select_from(DocumentPart)
            .where(
                DocumentPart.document_id == part.document_id,
                DocumentPart.order < part.order,
            )
        )
        return int(result.scalar_one()) + 1

    async def next_part_order(self, session: AsyncSession, document_id: UUID) -> int:
        await session.execute(
            select(Document.id).where(Document.id == document_id).with_for_update()
        )
        result = await session.execute(
            select(DocumentPart.order)
            .where(DocumentPart.document_id == document_id)
            .order_by(DocumentPart.order.desc())
            .limit(1)
        )
        current = result.scalar_one_or_none()
        if current is None:
            return 0
        return current + 1

    async def reorder_parts(
        self, session: AsyncSession, document: Document, ordered_part_ids: list[UUID]
    ) -> list[DocumentPart]:
        # ``populate_existing`` is what makes ``with_for_update`` mean anything here.
        # The caller reached this through ``require_document``, which eager-loads
        # ``Document.parts``, so every row is already in the session's identity map -
        # and SQLAlchemy returns the mapped instance untouched rather than overwriting
        # loaded attributes with the freshly locked values. The offset below is computed
        # from ``part.order``, so without this it could be computed from orders a
        # concurrent reorder had already superseded, land on the range that transaction
        # wrote, and violate uq_document_parts_document_order - a 500 on a plain drag.
        result = await session.execute(
            select(DocumentPart)
            .where(DocumentPart.document_id == document.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        current_parts = list(result.scalars().all())
        parts_by_id = {part.id: part for part in current_parts}
        if len(ordered_part_ids) != len(parts_by_id):
            return []
        if len(set(ordered_part_ids)) != len(ordered_part_ids):
            return []
        if set(ordered_part_ids) != set(parts_by_id):
            return []
        temporary_offset = temporary_reorder_offset(
            [part.order for part in current_parts], len(ordered_part_ids)
        )
        await session.execute(
            update(DocumentPart)
            .where(DocumentPart.document_id == document.id)
            .values(order=DocumentPart.order + temporary_offset)
        )
        for index, part_id in enumerate(ordered_part_ids):
            await session.execute(
                update(DocumentPart).where(DocumentPart.id == part_id).values(order=index)
            )
        await session.commit()
        result = await session.execute(
            select(DocumentPart)
            .where(DocumentPart.document_id == document.id)
            .order_by(DocumentPart.order)
        )
        return list(result.scalars().all())

    async def delete_part(self, session: AsyncSession, part: DocumentPart) -> None:
        # A pending row's key is the sentinel, not an object key: enqueueing it
        # verbatim would poison the GC with a key the store refuses forever. The
        # sentinel carries the minted key; a bare "pending" row has no blob at all.
        image_key = part.image_key
        if image_key.startswith("pending:"):
            image_key = image_key.removeprefix("pending:")
        elif image_key.startswith("pending"):
            image_key = ""
        if image_key:
            await self._add_media_deletion_intent(session, image_key)
        await session.delete(part)
        await session.commit()

    async def enqueue_media_deletion_intent(self, session: AsyncSession, image_key: str) -> None:
        await self._add_media_deletion_intent(session, image_key)
        await session.commit()

    async def _add_media_deletion_intent(self, session: AsyncSession, image_key: str) -> None:
        """Stage an intent unless one for this key is already queued.

        ``image_key`` is unique on the table, so a blind insert after a rejected
        finalize already queued the same key would fail the whole transaction.
        """
        existing = await session.execute(
            select(MediaDeletionIntent.id).where(MediaDeletionIntent.image_key == image_key)
        )
        if existing.scalar_one_or_none() is None:
            session.add(MediaDeletionIntent(image_key=image_key))
