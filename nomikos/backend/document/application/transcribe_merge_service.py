"""Transcribe merge - persist model transcription layer from ML output."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.document.infrastructure.orm_models import (
    DocumentPart,
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)
from nomikos_inference.contracts.transcribe import TranscribeRunResponse


class TranscribeJobHandlerError(Exception):
    """Raised for user-actionable transcribe job failures."""


class TranscribeMergeService:
    """Create a model transcription layer from per-line ML transcribe output."""

    @staticmethod
    def load_lines(session: Session, part_id: UUID) -> list[Line]:
        lines = list(
            session.execute(
                select(Line).where(Line.part_id == part_id).order_by(Line.order, Line.created_at)
            )
            .scalars()
            .all()
        )
        if not lines:
            raise TranscribeJobHandlerError(
                "Cannot transcribe a document part without layout lines"
            )
        return lines

    def apply_sync(
        self,
        session: Session,
        *,
        document_id: UUID,
        part_id: UUID,
        job_id: UUID | None,
        lines_with_output: list[tuple[Line, TranscribeRunResponse]],
        commit: bool = True,
        layer_name: str | None = None,
    ) -> dict:
        # Load the part locked, not with ``session.get``. Segment health decides a
        # merge or a deletion from a snapshot of which lines carry text, and
        # ``_has_text`` counts every layer, model output included: an unapproved
        # draft is the thing a reviewer is about to correct, which is why
        # ``test_an_unapproved_model_prediction_still_counts_as_text`` exists. A
        # layer committed inside that snapshot's window is therefore attached to a
        # line the apply is about to delete, and the draft goes with it. Holding
        # the part until this commits makes the two wait for each other.
        part = session.execute(
            select(DocumentPart).where(DocumentPart.id == part_id).with_for_update()
        ).scalar_one_or_none()
        if part is None or part.document_id != document_id:
            raise TranscribeJobHandlerError("Document part not found")

        resolved_name = layer_name or (
            f"Model transcription {job_id.hex[:8]}"
            if job_id is not None
            else "Local model transcription"
        )
        layer = Transcription(
            document_id=document_id,
            name=resolved_name,
            kind=TranscriptionKind.model,
            created_by_job_id=job_id,
        )
        session.add(layer)
        session.flush()

        result_lines: list[dict] = []
        for line, output in lines_with_output:
            session.add(
                LineTranscription(
                    line_id=line.id,
                    transcription_id=layer.id,
                    text=output.text,
                    confidence=output.confidence,
                )
            )
            result_lines.append(
                {
                    "line_id": str(line.id),
                    "text": output.text,
                    "confidence": output.confidence,
                }
            )

        if commit:
            session.commit()
        return {
            "transcription_id": str(layer.id),
            "lines": result_lines,
        }
