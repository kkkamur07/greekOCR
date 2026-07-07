"""Shared DTO builders for document line responses."""

from backend.document.api.schemas import LineResponse, LineTranscriptionResponse
from backend.document.infrastructure.orm_models import Line


def line_response(line: Line) -> LineResponse:
    return LineResponse(
        id=line.id,
        part_id=line.part_id,
        block_id=line.block_id,
        order=line.order,
        baseline=line.baseline,
        mask=line.mask,
        kind=line.kind,
        points=line.points,
        source=line.source,
        source_metadata=line.source_metadata,
        kraken_ceiling=line.kraken_ceiling,
        manual_geometry=line.manual_geometry,
        line_transcriptions=[
            LineTranscriptionResponse(
                id=transcription.id,
                transcription_id=transcription.transcription_id,
                transcription_kind=transcription.transcription.kind,
                text=transcription.text,
                confidence=transcription.confidence,
            )
            for transcription in line.transcriptions
        ],
        created_at=line.created_at,
    )
