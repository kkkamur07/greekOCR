"""Unit tests for line transcription text source domain semantics."""

from __future__ import annotations

from backend.document.domain.line_transcription_text import (
    LineTranscriptionTextSource,
    character_confidences_for_text,
    normalize_character_confidences,
)
from backend.document.infrastructure.orm_models import LineTranscription


def test_text_source_enum_values() -> None:
    assert LineTranscriptionTextSource.model.value == "model"
    assert LineTranscriptionTextSource.human_edited.value == "human_edited"


def test_character_confidences_for_text_returns_one_entry_per_character() -> None:
    result = character_confidences_for_text("ab", base_confidence=0.91)
    assert result == [
        {"char": "a", "confidence": 0.91},
        {"char": "b", "confidence": 0.91},
    ]


def test_character_confidences_for_text_returns_none_for_empty_text() -> None:
    assert character_confidences_for_text("") is None


def test_normalize_character_confidences_accepts_valid_adapter_shape() -> None:
    raw = [
        {"char": "a", "confidence": 0.9},
        {"char": "b", "confidence": 0.8},
    ]
    assert normalize_character_confidences("ab", raw) == raw


def test_normalize_character_confidences_falls_back_when_lengths_mismatch() -> None:
    raw = [{"char": "a", "confidence": 0.9}]
    result = normalize_character_confidences("ab", raw)
    assert result == [
        {"char": "a", "confidence": 0.9},
        {"char": "b", "confidence": 0.9},
    ]


def test_model_layer_row_keeps_character_confidences() -> None:
    row = LineTranscription(
        text="mock transcription 1",
        confidence=0.91,
        text_source=LineTranscriptionTextSource.model,
        character_confidences=character_confidences_for_text(
            "mock transcription 1",
            base_confidence=0.91,
        ),
    )
    assert row.text_source is LineTranscriptionTextSource.model
    assert row.character_confidences is not None
    assert len(row.character_confidences) == len(row.text)


def test_human_edited_row_clears_model_confidence_fields() -> None:
    row = LineTranscription(
        text="curated ground truth",
        confidence=0.91,
        text_source=LineTranscriptionTextSource.model,
        character_confidences=character_confidences_for_text("curated ground truth"),
    )
    row.text_source = LineTranscriptionTextSource.human_edited
    row.confidence = None
    row.character_confidences = None

    assert row.text_source is LineTranscriptionTextSource.human_edited
    assert row.confidence is None
    assert row.character_confidences is None
