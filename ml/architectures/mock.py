"""Deterministic transcribe runner for tests and local dev without weights."""

from __future__ import annotations

from ml.contracts.transcribe import CharacterConfidence, TranscribeRunResponse


class MockTranscribeRunner:
    def transcribe(self, image_bytes: bytes, *, params: dict) -> TranscribeRunResponse:
        del image_bytes
        line_index = int(params.get("line_index", 0))
        text = f"mock transcription {line_index + 1}"
        confidence = round(max(0.01, 0.91 - (line_index * 0.09)), 2)
        character_confidences = [
            CharacterConfidence(char=char, confidence=confidence) for char in text
        ]
        return TranscribeRunResponse(
            text=text,
            confidence=confidence,
            character_confidences=character_confidences,
        )
