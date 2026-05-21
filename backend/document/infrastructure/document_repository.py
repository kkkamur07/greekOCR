"""Document and DocumentPart persistence."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.document.infrastructure.orm_models import Document, DocumentPart, DocumentWorkflow


class DocumentRepository:
    async def get_by_id(self, session: AsyncSession, document_id: UUID) -> Document | None:
        result = await session.execute(
            select(Document)
            .options(selectinload(Document.parts))
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

    async def create(
        self,
        session: AsyncSession,
        *,
        project_id: UUID,
        name: str,
        workflow: DocumentWorkflow = DocumentWorkflow.draft,
    ) -> Document:
        document = Document(project_id=project_id, name=name, workflow=workflow)
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
        result = await session.execute(select(DocumentPart).where(DocumentPart.id == part_id))
        return result.scalar_one_or_none()

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
