"""Synchronous OCR prediction for pairing assist and page-level assist."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from uuid import UUID

import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.annotation.application.processing import apply_step
from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.document.application.document_service import DocumentService
from backend.document.domain.line_transcription_text import (
    LineTranscriptionTextSource,
    normalize_character_confidences,
)
from backend.document.infrastructure.orm_models import (
    DocumentPart,
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)
from backend.ml.application.model_service import InferenceModelService
from backend.ml.infrastructure.ml_client import MlServiceClient
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
from backend.users.infrastructure.orm_models import User


@dataclass(frozen=True)
class OcrPredictLineResult:
    line_id: UUID
    text: str
    confidence: float | None
    text_source: LineTranscriptionTextSource
    character_confidences: tuple[dict[str, object], ...] | None


@dataclass(frozen=True)
class OcrPredictResult:
    transcription_id: UUID
    transcription_name: str
    model_id: UUID
    model_name: str
    lines: list[OcrPredictLineResult]


def pairing_assist_layer_name(model_name: str) -> str:
    return f"Pairing assist ({model_name})"


class OcrPredictionService:
    def __init__(
        self,
        *,
        documents: DocumentService | None = None,
        inference: InferenceModelService | None = None,
        ml_client: MlServiceClient | None = None,
    ) -> None:
        self._documents = documents or DocumentService()
        self._inference = inference or InferenceModelService()
        self._ml = ml_client or MlServiceClient()

    async def predict_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        line_id: UUID,
        *,
        model_id: UUID | None = None,
    ) -> OcrPredictResult:
        part, model, lines = await self._prepare_prediction(
            session,
            user,
            project_id,
            document_id,
            part_id,
            model_id=model_id,
            line_ids=[line_id],
        )
        return await self._predict_lines(session, part, model, lines, line_index_offset=lines[0].order)

    async def predict_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        model_id: UUID | None = None,
    ) -> OcrPredictResult:
        part, model, lines = await self._prepare_prediction(
            session,
            user,
            project_id,
            document_id,
            part_id,
            model_id=model_id,
            line_ids=None,
        )
        return await self._predict_lines(session, part, model, lines, line_index_offset=0)

    async def _prepare_prediction(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        model_id: UUID | None,
        line_ids: list[UUID] | None,
    ) -> tuple[DocumentPart, InferenceModel, list[Line]]:
        await self._documents.get_document(session, user, project_id, document_id)
        part = await self._documents.get_part_for_media(session, user, part_id)
        if part.document_id != document_id:
            raise NotFoundError("Part not found")

        model = await self._resolve_model(
            session,
            user,
            project_id,
            document_id,
            part_id,
            model_id=model_id,
        )
        if model.task != InferenceTask.transcribe:
            raise ValidationError("OCR prediction requires a transcribe model")

        all_lines = await self._documents.list_part_lines(
            session, user, project_id, document_id, part_id
        )
        if not all_lines:
            raise ConflictError("Cannot run OCR prediction on a document part without layout lines")

        if line_ids is None:
            return part, model, all_lines

        lines_by_id = {line.id: line for line in all_lines}
        missing = [requested_id for requested_id in line_ids if requested_id not in lines_by_id]
        if missing:
            raise NotFoundError("Line not found")
        return part, model, [lines_by_id[requested_id] for requested_id in line_ids]

    async def _resolve_model(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        model_id: UUID | None,
    ) -> InferenceModel:
        if model_id is not None:
            return await self._inference._require_model_for_task(  # noqa: SLF001
                session,
                model_id,
                InferenceTask.transcribe,
            )
        binding = await self._inference.resolve_for_part(
            session,
            user,
            project_id,
            document_id,
            part_id,
            task=InferenceTask.transcribe,
        )
        return binding.model

    async def _predict_lines(
        self,
        session: AsyncSession,
        part: DocumentPart,
        model: InferenceModel,
        lines: list[Line],
        *,
        line_index_offset: int,
    ) -> OcrPredictResult:
        page_image = self._load_page_image(part)
        layer = await self._get_or_create_pairing_assist_layer(session, part.document_id, model.name)
        effective_params = dict(model.default_params or {})

        results: list[OcrPredictLineResult] = []
        for index, line in enumerate(lines):
            image_bytes = self._rectify_line_image(page_image, line)
            ml_result = await self._ml.run_transcribe(
                registry_model_id=model.registry_model_id,
                registry_tag=model.registry_tag,
                image_bytes=image_bytes,
                params={**effective_params, "line_index": line_index_offset + index},
            )
            character_confidences = normalize_character_confidences(
                ml_result.text,
                [entry.model_dump() for entry in ml_result.character_confidences],
                base_confidence=ml_result.confidence,
            )
            await self._upsert_line_transcription(
                session,
                layer=layer,
                line=line,
                text=ml_result.text,
                confidence=ml_result.confidence,
                character_confidences=character_confidences,
            )
            results.append(
                OcrPredictLineResult(
                    line_id=line.id,
                    text=ml_result.text,
                    confidence=ml_result.confidence,
                    text_source=LineTranscriptionTextSource.model,
                    character_confidences=tuple(character_confidences) if character_confidences else None,
                )
            )

        await session.commit()
        await session.refresh(layer)
        return OcrPredictResult(
            transcription_id=layer.id,
            transcription_name=layer.name,
            model_id=model.id,
            model_name=model.name,
            lines=results,
        )

    async def _get_or_create_pairing_assist_layer(
        self,
        session: AsyncSession,
        document_id: UUID,
        model_name: str,
    ) -> Transcription:
        layer_name = pairing_assist_layer_name(model_name)
        result = await session.execute(
            select(Transcription).where(
                Transcription.document_id == document_id,
                Transcription.kind == TranscriptionKind.model,
                Transcription.name == layer_name,
            )
        )
        layer = result.scalar_one_or_none()
        if layer is None:
            layer = Transcription(
                document_id=document_id,
                name=layer_name,
                kind=TranscriptionKind.model,
            )
            session.add(layer)
            await session.flush()
        return layer

    async def _upsert_line_transcription(
        self,
        session: AsyncSession,
        *,
        layer: Transcription,
        line: Line,
        text: str,
        confidence: float | None,
        character_confidences: list[dict[str, object]] | None,
    ) -> LineTranscription:
        result = await session.execute(
            select(LineTranscription)
            .where(
                LineTranscription.transcription_id == layer.id,
                LineTranscription.line_id == line.id,
            )
            .options(selectinload(LineTranscription.transcription))
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = LineTranscription(
                line_id=line.id,
                transcription_id=layer.id,
                text=text,
                confidence=confidence,
                text_source=LineTranscriptionTextSource.model,
                character_confidences=character_confidences,
            )
            session.add(row)
            return row

        row.text = text
        row.confidence = confidence
        row.text_source = LineTranscriptionTextSource.model
        row.character_confidences = character_confidences
        return row

    def _load_page_image(self, part: DocumentPart) -> np.ndarray:
        image_bytes = self._documents.read_part_bytes(part)
        with Image.open(BytesIO(image_bytes)) as image:
            return np.array(image.convert("RGB"))

    @staticmethod
    def _rectify_line_image(page_image: np.ndarray, line: Line) -> bytes:
        segment = {"points": line.points, "kind": line.kind.value}
        rectified = apply_step(page_image, segment, "rectify")
        buffer = BytesIO()
        Image.fromarray(rectified).save(buffer, format="PNG")
        return buffer.getvalue()
