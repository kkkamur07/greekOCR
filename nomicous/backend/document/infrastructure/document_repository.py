"""Document and DocumentPart persistence."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    DocumentWorkflow,
    Line,
    LineTranscription,
    PageTranscriptionLine,
    Transcription,
    TranscriptionKind,
)


class DocumentRepository:
    async def get_by_id(self, session: AsyncSession, document_id: UUID) -> Document | None:
        result = await session.execute(
            select(Document)
            .options(selectinload(Document.parts))
            .options(selectinload(Document.transcriptions))
            .where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_for_project(
        self,
        session: AsyncSession,
        project_id: UUID,
        *,
        include_archived: bool = False,
    ) -> list[Document]:
        stmt = select(Document).where(Document.project_id == project_id)
        if not include_archived:
            stmt = stmt.where(Document.workflow != DocumentWorkflow.archived)
        stmt = stmt.order_by(Document.created_at.desc())
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

    async def list_page_transcription_lines(
        self, session: AsyncSession, part_id: UUID
    ) -> list[PageTranscriptionLine]:
        result = await session.execute(
            select(PageTranscriptionLine)
            .where(PageTranscriptionLine.part_id == part_id)
            .order_by(PageTranscriptionLine.order, PageTranscriptionLine.created_at)
        )
        return list(result.scalars().all())

    async def next_part_order(self, session: AsyncSession, document_id: UUID) -> int:
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

    async def add_part(
        self,
        session: AsyncSession,
        *,
        document_id: UUID,
        image_key: str,
        order: int,
        width: int | None = None,
        height: int | None = None,
    ) -> DocumentPart:
        part = DocumentPart(
            document_id=document_id,
            image_key=image_key,
            order=order,
            width=width,
            height=height,
        )
        session.add(part)
        await session.commit()
        await session.refresh(part)
        return part

    async def reorder_parts(
        self, session: AsyncSession, document: Document, ordered_part_ids: list[UUID]
    ) -> list[DocumentPart]:
        parts_by_id = {p.id: p for p in document.parts}
        if len(ordered_part_ids) != len(parts_by_id):
            return []
        if len(set(ordered_part_ids)) != len(ordered_part_ids):
            return []
        if set(ordered_part_ids) != set(parts_by_id):
            return []
        for index, part_id in enumerate(ordered_part_ids):
            parts_by_id[part_id].order = index
        await session.commit()
        result = await session.execute(
            select(DocumentPart)
            .where(DocumentPart.document_id == document.id)
            .order_by(DocumentPart.order)
        )
        return list(result.scalars().all())

    async def delete_part(self, session: AsyncSession, part: DocumentPart) -> None:
        await session.delete(part)
        await session.commit()
