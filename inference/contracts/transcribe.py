"""Transcribe task request/response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class CharacterConfidence(BaseModel):
    char: str = Field(min_length=1, max_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class TranscribeRunResponse(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    character_confidences: list[CharacterConfidence]

    @model_validator(mode="after")
    def align_character_confidences_with_text(self) -> TranscribeRunResponse:
        if len(self.character_confidences) != len(self.text):
            raise ValueError("character_confidences length must match text length")

        for index, entry in enumerate(self.character_confidences):
            if entry.char != self.text[index]:
                raise ValueError(f"character_confidences[{index}].char must match text[{index}]")
        return self


class TranscribeLineRegion(BaseModel):
    line_id: str | None = None
    line_index: int = Field(ge=0)
    points: list[list[float]] | None = None


class TranscribeBatchLineResult(BaseModel):
    line_id: str | None = None
    line_index: int = Field(ge=0)
    output: TranscribeRunResponse


class TranscribeBatchRunResponse(BaseModel):
    lines: list[TranscribeBatchLineResult] = Field(min_length=1)
