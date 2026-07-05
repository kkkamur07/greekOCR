"""Document and part use cases with project membership checks."""

from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.annotation.application.line_geometry import resolve_line_baseline_and_mask
from backend.core.exceptions import AccessDeniedError, ConflictError, NotFoundError, ValidationError
from backend.core.settings.job import get_job_settings
from backend.document.domain.access import require_can_read
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import MediaStore
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    DocumentWorkflow,
    Line,
    LineGeometryKind,
    LineSource,
    LineTranscription,
    PageTranscriptionLine,
    Transcription,
    TranscriptionKind,
)
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.ml.application.model_service import InferenceModelService
from backend.ml.infrastructure.orm_models import InferenceTask
from backend.project.domain.access import is_member
from backend.project.infrastructure.orm_models import Project
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User

MAX_PAGE_TRANSCRIPTION_LINES = 10_000
MAX_REPLACE_PART_LINES = 10_000

DOCUMENT_UPDATE_FIELDS = frozenset({"name", "workflow"})
BLOCK_PATCH_FIELDS = frozenset({"order", "box"})
LINE_PATCH_FIELDS = frozenset({"order", "block_id", "baseline", "mask", "points"})


class DocumentService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        media: MediaStore | None = None,
        inference_models: InferenceModelService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._media = media or MediaStore()
        self._inference_models = inference_models or InferenceModelService()
        self._job_settings = get_job_settings()

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

    async def list_parts(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
    ) -> list[DocumentPart]:
        document = await self.get_document(session, user, project_id, document_id)
        return sorted(document.parts, key=lambda p: p.order)

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
        suffix = "bin"
        filename_stem: str | None = None
        if filename and "." in filename:
            path = Path(filename)
            suffix = path.suffix.lstrip(".").lower()[:16]
            filename_stem = path.stem
        part = DocumentPart(document_id=document.id, order=order, image_key="pending")
        session.add(part)
        await session.flush()
        image_key = self._media.part_image_key(part.id, suffix=suffix, filename_stem=filename_stem)
        try:
            self._media.write(image_key, data)
            part.image_key = image_key
            await session.commit()
        except Exception:
            await session.rollback()
            self._media.delete(image_key)
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

    async def list_part_lines(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> list[Line]:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        return await self._documents.list_part_lines(session, part.id)

    async def list_part_layout(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> tuple[list[Block], list[Line]]:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        blocks = await self._list_part_blocks(session, part.id)
        lines = await self._documents.list_part_lines(session, part.id)
        return blocks, lines

    async def create_part_block(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        order: int,
        box: dict,
    ) -> Block:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        block = Block(part_id=part.id, order=order, box=box, manual_geometry=True)
        session.add(block)
        await session.commit()
        await session.refresh(block)
        return block

    async def patch_part_block(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        block_id: UUID,
        **updates: object,
    ) -> Block:
        self._reject_unknown_fields(updates, BLOCK_PATCH_FIELDS, "block patch")
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        block = await self._block_or_404(session, part.id, block_id)
        for key, value in updates.items():
            if value is not None:
                setattr(block, key, value)
        block.manual_geometry = True
        await session.commit()
        await session.refresh(block)
        return block

    async def delete_part_block(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        block_id: UUID,
    ) -> None:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        block = await self._block_or_404(session, part.id, block_id)
        await session.delete(block)
        await session.commit()

    async def create_part_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        order: int,
        kind: LineGeometryKind,
        points: list[list[float]],
        block_id: UUID | None = None,
        baseline: dict | None = None,
        mask: dict | None = None,
    ) -> Line:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        if block_id is not None:
            await self._block_or_404(session, part.id, block_id)
        line = Line(
            part_id=part.id,
            block_id=block_id,
            order=order,
            kind=kind,
            points=points,
            baseline=baseline or {"points": points},
            mask=mask or {"points": points},
            source=LineSource.manual,
            manual_geometry=True,
        )
        session.add(line)
        await session.commit()
        return await self._line_or_404(session, part.id, line.id)

    async def patch_part_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        line_id: UUID,
        **updates: object,
    ) -> Line:
        self._reject_unknown_fields(updates, LINE_PATCH_FIELDS, "line patch")
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        line = await self._line_or_404(session, part.id, line_id)
        if "block_id" in updates and updates["block_id"] is not None:
            await self._block_or_404(session, part.id, updates["block_id"])
        for key, value in updates.items():
            if value is not None:
                setattr(line, key, value)
        line.manual_geometry = True
        line.source = LineSource.manual
        await session.commit()
        return await self._line_or_404(session, part.id, line.id)

    async def delete_part_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        line_id: UUID,
    ) -> None:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        line = await self._line_or_404(session, part.id, line_id)
        await session.delete(line)
        await session.commit()

    async def reset_part_layout(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        line_ids: list[UUID] | None = None,
    ) -> tuple[list[Block], list[Line]]:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        lines = await self._documents.list_part_lines(session, part.id)
        selected_ids = set(line_ids) if line_ids is not None else {line.id for line in lines}
        if line_ids is not None and selected_ids - {line.id for line in lines}:
            raise NotFoundError("Line not found")
        for line in lines:
            if line.id in selected_ids:
                line.manual_geometry = False
        if line_ids is None:
            blocks = await self._list_part_blocks(session, part.id)
            for block in blocks:
                block.manual_geometry = False
        await session.commit()
        return await self.list_part_layout(session, user, project_id, document_id, part_id)

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

    async def replace_part_lines(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        lines: list[dict],
        allow_new_ids: bool = False,
    ) -> list[Line]:
        if len(lines) > MAX_REPLACE_PART_LINES:
            raise ValidationError(
                f"Cannot replace more than {MAX_REPLACE_PART_LINES} lines at once"
            )
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        ground_truth = await self._ensure_ground_truth_transcription(session, document)

        requested_ids = [line["id"] for line in lines if line.get("id") is not None]
        if len(set(requested_ids)) != len(requested_ids):
            raise ValidationError("Line ids must be unique")

        existing = await self._documents.list_part_lines(session, part.id)
        existing_by_id = {line.id: line for line in existing}
        if not allow_new_ids:
            new_supplied_ids = [
                line_id for line_id in requested_ids if line_id not in existing_by_id
            ]
            if new_supplied_ids:
                raise ValidationError("New line ids are server-generated")
        requested_id_set = set(requested_ids)
        for line in existing:
            if line.id not in requested_id_set:
                await session.delete(line)

        for data in lines:
            line_id = data.get("id")
            prior = existing_by_id.get(line_id) if line_id is not None else None
            line = prior
            if line is None:
                line = Line(part_id=part.id, baseline={}, mask=None)
                if line_id is not None:
                    line.id = line_id
                session.add(line)

            points = data["points"]
            block_id = data.get("block_id")
            if block_id is not None:
                await self._block_or_404(session, part.id, block_id)
            line.block_id = block_id
            line.order = data["order"]
            line.kind = data["kind"]
            line.points = points
            line.baseline, line.mask = resolve_line_baseline_and_mask(
                points=points,
                payload_baseline=data.get("baseline"),
                payload_mask=data.get("mask"),
                existing_baseline=prior.baseline if prior is not None else None,
                existing_mask=prior.mask if prior is not None else None,
            )
            line.source = data["source"]
            line.source_metadata = data.get("source_metadata")
            line.kraken_ceiling = data.get("kraken_ceiling")
            source_value = (
                data["source"].value if hasattr(data["source"], "value") else data["source"]
            )
            line.manual_geometry = source_value == "manual"

            await self._set_ground_truth_text(
                line, ground_truth, data.get("approved_text"), session
            )

        await session.commit()
        return await self._documents.list_part_lines(session, part.id)

    async def enqueue_transcribe_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        model_id: UUID | None = None,
        line_ids: list[UUID] | None = None,
    ) -> Job:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        lines = await self._documents.list_part_lines(session, part.id)
        if not lines:
            raise ConflictError("Cannot transcribe a document part without layout lines")
        if line_ids is not None:
            selected_ids = set(line_ids)
            known_ids = {line.id for line in lines}
            if selected_ids - known_ids:
                raise NotFoundError("Line not found")
            if not selected_ids:
                raise ValidationError("At least one line must be selected for transcription")
        binding_id: UUID | None = None
        selected_model_id = model_id
        ml_params: dict = {}
        if selected_model_id is not None:
            model = await self._inference_models.get_model_for_task(
                session, selected_model_id, InferenceTask.transcribe
            )
            ml_params = dict(model.default_params or {})
        else:
            try:
                resolved = await self._inference_models.resolve_for_part(
                    session,
                    user,
                    project_id,
                    document_id,
                    part_id,
                    task=InferenceTask.transcribe,
                )
            except NotFoundError:
                selected_model_id = None
            else:
                selected_model_id = resolved.model.id
                binding_id = resolved.binding.id
                ml_params = dict(resolved.effective_params)
        payload: dict = {"adapter": self._job_settings.transcribe_adapter, "ml_params": ml_params}
        if line_ids is not None:
            payload["line_ids"] = [str(line_id) for line_id in line_ids]
        job = Job(
            type=JobType.transcribe,
            status=JobStatus.pending,
            user_id=user.id,
            document_id=document.id,
            document_part_id=part.id,
            model_id=selected_model_id,
            binding_id=binding_id,
            payload=payload,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job

    async def enqueue_segment_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        ml_params: dict | None = None,
    ) -> Job:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        job = Job(
            type=JobType.segment,
            status=JobStatus.pending,
            user_id=user.id,
            document_id=document.id,
            document_part_id=part.id,
            payload={
                "adapter": self._job_settings.segment_adapter,
                "ml_params": ml_params or {},
            },
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job

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
        image_key = part.image_key
        self._media.delete(image_key)
        await self._documents.delete_part(session, part)

    async def get_part_for_media(
        self,
        session: AsyncSession,
        user: User,
        part_id: UUID,
    ) -> DocumentPart:
        part = await self._documents.get_part(session, part_id)
        if part is None:
            raise NotFoundError("Part not found")
        document = await self._documents.get_by_id(session, part.document_id)
        if document is None:
            raise NotFoundError("Document not found")
        await self._require_member(session, document.project_id, user.id)
        return part

    async def get_part_for_public_media(
        self,
        session: AsyncSession,
        part_id: UUID,
    ) -> DocumentPart:
        part = await self._documents.get_part(session, part_id)
        if part is None:
            raise NotFoundError("Part not found")
        document = await self._documents.get_by_id(session, part.document_id)
        if document is None:
            raise NotFoundError("Document not found")
        project = await self._load_project(session, document.project_id)
        require_can_read(document, project, None)
        return part

    def resolve_part_image_path(self, part: DocumentPart) -> Path:
        """Resolved on-disk path for streaming; raises NotFoundError if missing."""
        try:
            path = self._media.absolute_path(part.image_key)
        except (ValueError, FileNotFoundError):
            raise NotFoundError("Part image not found") from None
        if not path.is_file():
            raise NotFoundError("Part image not found")
        return path

    def read_part_bytes(self, part: DocumentPart) -> bytes:
        try:
            return self._media.read(part.image_key)
        except (ValueError, FileNotFoundError):
            raise NotFoundError("Part image not found") from None

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

    async def _block_or_404(
        self, session: AsyncSession, part_id: UUID, block_id: UUID
    ) -> Block:
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
            .options(selectinload(Line.transcriptions).selectinload(LineTranscription.transcription))
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
