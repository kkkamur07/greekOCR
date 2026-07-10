"""Shared constants and private helpers for document application services."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.core.exceptions import AccessDeniedError, NotFoundError, ValidationError
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import MediaStore
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    Line,
    LineTranscription,
    PageTranscriptionLine,
    Transcription,
    TranscriptionKind,
)
from backend.ml.application.model_service import InferenceModelService
from backend.project.domain.access import is_member
from backend.project.infrastructure.orm_models import Project
from backend.project.infrastructure.project_repository import ProjectRepository

MAX_PAGE_TRANSCRIPTION_LINES = 10_000
MAX_REPLACE_PART_LINES = 10_000

DOCUMENT_UPDATE_FIELDS = frozenset({"name", "workflow"})
BLOCK_PATCH_FIELDS = frozenset({"order", "box"})
LINE_PATCH_FIELDS = frozenset({"order", "block_id", "baseline", "mask", "points"})


class DocumentServiceSharedMixin:
    _documents: DocumentRepository
    _projects: ProjectRepository
    _media: MediaStore
    _inference_models: InferenceModelService

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

    async def _document_part_or_404(
        self, session: AsyncSession, document: Document, part_id: UUID
    ) -> DocumentPart:
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document.id:
            raise NotFoundError("Part not found")
        return part

    async def _list_part_blocks(self, session: AsyncSession, part_id: UUID) -> list[Block]:
        result = await session.execute(
            select(Block).where(Block.part_id == part_id).order_by(Block.order, Block.created_at)
        )
        return list(result.scalars().all())

    async def _list_page_transcription_lines(
        self, session: AsyncSession, part_id: UUID
    ) -> list[PageTranscriptionLine]:
        return await self._documents.list_page_transcription_lines(session, part_id)

    async def _page_transcription_line_or_404(
        self, session: AsyncSession, part_id: UUID, order: int
    ) -> PageTranscriptionLine:
        result = await session.execute(
            select(PageTranscriptionLine).where(
                PageTranscriptionLine.part_id == part_id,
                PageTranscriptionLine.order == order,
            )
        )
        text_line = result.scalar_one_or_none()
        if text_line is None:
            raise NotFoundError("Text line not found")
        return text_line

    def _split_page_transcription(self, text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    @staticmethod
    def _reject_unknown_fields(
        fields: dict[str, object], allowed: frozenset[str], operation: str
    ) -> None:
        unknown = set(fields) - allowed
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValidationError(f"Unsupported {operation} field(s): {joined}")

    async def _block_or_404(self, session: AsyncSession, part_id: UUID, block_id: UUID) -> Block:
        result = await session.execute(
            select(Block).where(Block.id == block_id, Block.part_id == part_id)
        )
        block = result.scalar_one_or_none()
        if block is None:
            raise NotFoundError("Block not found")
        return block

    async def _line_or_404(self, session: AsyncSession, part_id: UUID, line_id: UUID) -> Line:
        result = await session.execute(
            select(Line)
            .where(Line.id == line_id, Line.part_id == part_id)
            .options(
                selectinload(Line.transcriptions).selectinload(LineTranscription.transcription)
            )
        )
        line = result.scalar_one_or_none()
        if line is not None:
            return line
        raise NotFoundError("Line not found")

    async def _count_part_lines(self, session: AsyncSession, part_id: UUID) -> int:
        result = await session.execute(
            select(func.count()).select_from(Line).where(Line.part_id == part_id)
        )
        return int(result.scalar_one())

    async def _count_paired_ground_truth_lines(self, session: AsyncSession, part_id: UUID) -> int:
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

    async def _transcription_or_404(
        self, session: AsyncSession, document: Document, transcription_id: UUID
    ) -> Transcription:
        transcription = await session.get(Transcription, transcription_id)
        if transcription is None or transcription.document_id != document.id:
            raise NotFoundError("Transcription layer not found")
        return transcription

    async def _line_in_document_or_404(
        self, session: AsyncSession, document: Document, line_id: UUID
    ) -> Line:
        result = await session.execute(
            select(Line)
            .join(DocumentPart, Line.part_id == DocumentPart.id)
            .where(Line.id == line_id, DocumentPart.document_id == document.id)
        )
        line = result.scalar_one_or_none()
        if line is None:
            raise NotFoundError("Line not found")
        return line

    async def _ensure_ground_truth_transcription(
        self, session: AsyncSession, document: Document
    ) -> Transcription:
        transcription = await self._documents.get_ground_truth_transcription(session, document.id)
        if transcription is not None:
            return transcription
        transcription = Transcription(
            document_id=document.id,
            name="Ground truth",
            kind=TranscriptionKind.ground_truth,
        )
        session.add(transcription)
        await session.flush()
        return transcription

    async def _set_ground_truth_text(
        self,
        line: Line,
        ground_truth: Transcription,
        text: str | None,
        session: AsyncSession,
    ) -> None:
        existing = next(
            (
                transcription
                for transcription in line.transcriptions
                if transcription.transcription_id == ground_truth.id
            ),
            None,
        )
        if text is None:
            if existing is not None:
                line.transcriptions.remove(existing)
                await session.delete(existing)
            return
        if existing is None:
            line.transcriptions.append(
                LineTranscription(
                    transcription=ground_truth,
                    text=text,
                    confidence=None,
                )
            )
            return
        existing.text = text
        existing.confidence = None
