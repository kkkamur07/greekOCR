"""Shared DTO builders for line transcription rows."""

from backend.document.api.schemas import CharacterConfidenceResponse, LineTranscriptionResponse
from backend.document.infrastructure.orm_models import LineTranscription, TranscriptionKind


def character_confidence_responses(
    character_confidences: list[dict[str, object]] | None,
) -> list[CharacterConfidenceResponse] | None:
    if character_confidences is None:
        return None
    return [
        CharacterConfidenceResponse(char=str(item["char"]), confidence=float(item["confidence"]))
        for item in character_confidences
        if "char" in item and "confidence" in item
    ]


def line_transcription_response(
    line_transcription: LineTranscription,
    *,
    transcription_kind: TranscriptionKind | None = None,
) -> LineTranscriptionResponse:
    kind = transcription_kind
    if kind is None:
        kind = line_transcription.transcription.kind
    return LineTranscriptionResponse(
        id=line_transcription.id,
        transcription_id=line_transcription.transcription_id,
        transcription_kind=kind,
        text=line_transcription.text,
        confidence=line_transcription.confidence,
        text_source=line_transcription.text_source,
        character_confidences=character_confidence_responses(
            line_transcription.character_confidences
        ),
    )
