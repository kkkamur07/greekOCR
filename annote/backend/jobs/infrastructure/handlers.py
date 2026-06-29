"""Test job handlers — noop sleep and intentional failure."""

from __future__ import annotations

import time

from sqlalchemy import select

from backend.document.infrastructure.orm_models import (
    DocumentPart,
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)
from backend.jobs.infrastructure.orm_models import Job
from infrastructure.db import SyncSessionLocal


class TestJobHandlerError(Exception):
    """Raised by the fail test handler to exercise error persistence."""


class TranscribeJobHandlerError(Exception):
    """Raised for user-actionable transcribe job failures."""


def run_test_handler(job: Job) -> dict:
    handler = (job.payload or {}).get("handler", "noop")
    if handler == "fail":
        raise TestJobHandlerError("intentional test failure")
    time.sleep(0.1)
    return {"ok": True}


def run_transcribe_handler(job: Job) -> dict:
    """Persist a canonical mock transcribe result as a fresh model layer."""
    if job.document_id is None or job.document_part_id is None:
        raise TranscribeJobHandlerError("Transcribe job is missing its target document part")

    with SyncSessionLocal() as session:
        part = session.get(DocumentPart, job.document_part_id)
        if part is None or part.document_id != job.document_id:
            raise TranscribeJobHandlerError("Document part not found")

        lines = list(
            session.execute(
                select(Line)
                .where(Line.part_id == part.id)
                .order_by(Line.order, Line.created_at)
            )
            .scalars()
            .all()
        )
        if not lines:
            raise TranscribeJobHandlerError("Cannot transcribe a document part without layout lines")

        layer = Transcription(
            document_id=job.document_id,
            name=f"Model transcription {job.id.hex[:8]}",
            kind=TranscriptionKind.model,
            created_by_job_id=job.id,
        )
        session.add(layer)
        session.flush()

        result_lines: list[dict] = []
        for index, line in enumerate(lines):
            text = f"mock transcription {index + 1}"
            confidence = round(max(0.01, 0.91 - (index * 0.09)), 2)
            session.add(
                LineTranscription(
                    line_id=line.id,
                    transcription_id=layer.id,
                    text=text,
                    confidence=confidence,
                )
            )
            result_lines.append(
                {
                    "line_id": str(line.id),
                    "text": text,
                    "confidence": confidence,
                }
            )

        session.commit()
        return {
            "transcription_id": str(layer.id),
            "lines": result_lines,
        }
