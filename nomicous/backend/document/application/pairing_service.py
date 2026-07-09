"""Page transcription import and line pairing."""

from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.document.infrastructure.orm_models import PageTranscriptionLine
from backend.document.application.document_service_shared import (
    MAX_PAGE_TRANSCRIPTION_LINES,
    DocumentServiceSharedMixin,
)
from backend.users.infrastructure.orm_models import User


class PairingServiceMixin(DocumentServiceSharedMixin):
    async def import_page_transcription(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        text: str,
    ) -> tuple[list[PageTranscriptionLine], dict[str, int]]:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        text_lines = self._split_page_transcription(text)
        if len(text_lines) > MAX_PAGE_TRANSCRIPTION_LINES:
            raise ValidationError(
                f"Page transcription cannot exceed {MAX_PAGE_TRANSCRIPTION_LINES} non-empty lines"
            )
        existing = await self._list_page_transcription_lines(session, part.id)
        paired_line_ids = {
            text_line.paired_line_id for text_line in existing if text_line.paired_line_id
        }
        if paired_line_ids:
            ground_truth = await self._documents.get_ground_truth_transcription(
                session, document.id
            )
            if ground_truth is not None:
                for paired_line_id in paired_line_ids:
                    paired_line = await self._line_or_404(session, part.id, paired_line_id)
                    await self._set_ground_truth_text(paired_line, ground_truth, None, session)
        for text_line in existing:
            await session.delete(text_line)
        await session.flush()
        for order, line_text in enumerate(text_lines):
            session.add(PageTranscriptionLine(part_id=part.id, order=order, text=line_text))
        await session.commit()
        return await self.get_page_pairing(session, user, project_id, document_id, part_id)

    async def get_page_pairing(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> tuple[list[PageTranscriptionLine], dict[str, int]]:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        text_lines = await self._list_page_transcription_lines(session, part.id)
        total_lines = await self._count_part_lines(session, part.id)
        paired_lines = await self._count_paired_ground_truth_lines(session, part.id)
        percent = round((paired_lines / total_lines) * 100) if total_lines else 0
        return text_lines, {
            "paired_lines": paired_lines,
            "total_lines": total_lines,
            "percent": percent,
        }

    async def pair_page_text_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        line_id: UUID,
        text_line_order: int,
    ) -> tuple[list[PageTranscriptionLine], dict[str, int]]:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        line = await self._line_or_404(session, part.id, line_id)
        text_line = await self._page_transcription_line_or_404(
            session, part.id, text_line_order
        )
        ground_truth = await self._ensure_ground_truth_transcription(session, document)

        previous_paired_line_id = text_line.paired_line_id
        if previous_paired_line_id is not None and previous_paired_line_id != line.id:
            previous_line = await self._line_or_404(session, part.id, previous_paired_line_id)
            await self._set_ground_truth_text(previous_line, ground_truth, None, session)
        for candidate in await self._list_page_transcription_lines(session, part.id):
            if candidate.paired_line_id == line.id:
                candidate.paired_line_id = None
        text_line.paired_line_id = line.id
        await self._set_ground_truth_text(line, ground_truth, text_line.text, session)
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise ConflictError("This segment is already paired to another text line") from exc
        return await self.get_page_pairing(session, user, project_id, document_id, part_id)
