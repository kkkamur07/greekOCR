"""Ground truth transcription copy and edit use cases."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError
from backend.document.infrastructure.orm_models import (
    DocumentPart,
    Line,
    LineTranscription,
    TranscriptionKind,
)
from backend.document.application.document_service_shared import DocumentServiceSharedMixin
from backend.users.infrastructure.orm_models import User


class TranscriptionServiceMixin(DocumentServiceSharedMixin):
    async def copy_to_ground_truth(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        source_transcription_id: UUID,
        *,
        line_ids: list[UUID] | None = None,
    ) -> list[UUID]:
        document = await self.get_document(session, user, project_id, document_id)
        source = await self._transcription_or_404(session, document, source_transcription_id)
        if source.kind != TranscriptionKind.model:
            raise ConflictError("Copy to ground truth requires a model transcription layer")

        ground_truth = await self._ensure_ground_truth_transcription(session, document)
        stmt = (
            select(LineTranscription)
            .join(Line, LineTranscription.line_id == Line.id)
            .join(DocumentPart, Line.part_id == DocumentPart.id)
            .where(
                LineTranscription.transcription_id == source.id,
                DocumentPart.document_id == document.id,
            )
            .order_by(Line.order, Line.created_at)
        )
        if line_ids is not None:
            stmt = stmt.where(LineTranscription.line_id.in_(line_ids))
        result = await session.execute(stmt)
        source_rows = list(result.scalars().all())
        copied_line_ids = [row.line_id for row in source_rows]

        existing_by_line: dict[UUID, LineTranscription] = {}
        if copied_line_ids:
            existing_result = await session.execute(
                select(LineTranscription).where(
                    LineTranscription.transcription_id == ground_truth.id,
                    LineTranscription.line_id.in_(copied_line_ids),
                )
            )
            existing_by_line = {
                row.line_id: row for row in existing_result.scalars().all()
            }

        for source_row in source_rows:
            target = existing_by_line.get(source_row.line_id)
            if target is None:
                session.add(
                    LineTranscription(
                        line_id=source_row.line_id,
                        transcription_id=ground_truth.id,
                        text=source_row.text,
                        confidence=None,
                    )
                )
            else:
                target.text = source_row.text
                target.confidence = None

        await session.commit()
        return copied_line_ids

    async def patch_ground_truth_line_text(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        transcription_id: UUID,
        line_id: UUID,
        *,
        text: str,
    ) -> LineTranscription:
        document = await self.get_document(session, user, project_id, document_id)
        transcription = await self._transcription_or_404(session, document, transcription_id)
        if transcription.kind != TranscriptionKind.ground_truth:
            raise ConflictError("Only Ground truth transcription lines can be edited")
        await self._line_in_document_or_404(session, document, line_id)

        result = await session.execute(
            select(LineTranscription).where(
                LineTranscription.transcription_id == transcription.id,
                LineTranscription.line_id == line_id,
            )
        )
        line_transcription = result.scalar_one_or_none()
        if line_transcription is None:
            line_transcription = LineTranscription(
                line_id=line_id,
                transcription_id=transcription.id,
                text=text,
                confidence=None,
            )
            session.add(line_transcription)
        else:
            line_transcription.text = text
            line_transcription.confidence = None
        await session.commit()
        await session.refresh(line_transcription)
        return line_transcription
