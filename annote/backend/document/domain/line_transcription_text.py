"""Line transcription text source and per-character confidence helpers."""

from __future__ import annotations

import enum
from typing import TypedDict


class LineTranscriptionTextSource(str, enum.Enum):
    model = "model"
    human_edited = "human_edited"


class CharacterConfidence(TypedDict):
    char: str
    confidence: float


def character_confidences_for_text(
    text: str,
    *,
    base_confidence: float = 0.5,
) -> list[CharacterConfidence] | None:
    if not text:
        return None
    clamped = max(0.0, min(1.0, float(base_confidence)))
    return [{"char": char, "confidence": clamped} for char in text]


def _parse_character_confidence_entry(entry: object) -> CharacterConfidence | None:
    if not isinstance(entry, dict):
        return None
    char = entry.get("char")
    if char is None:
        char = entry.get("character")
    confidence = entry.get("confidence")
    if confidence is None:
        confidence = entry.get("probability")
    if not isinstance(char, str) or char == "" or confidence is None:
        return None
    try:
        return {"char": char, "confidence": float(confidence)}
    except (TypeError, ValueError):
        return None


def normalize_character_confidences(
    text: str,
    raw: list[object],
    *,
    base_confidence: float | None = None,
) -> list[CharacterConfidence] | None:
    if not text:
        return None

    entries: list[CharacterConfidence] = []
    for item in raw:
        parsed = _parse_character_confidence_entry(item)
        if parsed is not None:
            entries.append(parsed)

    if (
        entries
        and len(entries) == len(text)
        and "".join(item["char"] for item in entries) == text
    ):
        return entries

    fallback_base = base_confidence if base_confidence is not None else 0.5
    if entries:
        fallback_base = sum(item["confidence"] for item in entries) / len(entries)
    return character_confidences_for_text(text, base_confidence=fallback_base)
