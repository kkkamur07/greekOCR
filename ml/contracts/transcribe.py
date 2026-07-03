"""Transcribe task request/response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from ml.contracts.common import ImageBytes


class TranscribeRunRequest(BaseModel):
    registry_model_id: str = Field(min_length=1)
    registry_tag: str = Field(default="stable", min_length=1)
    image_bytes: ImageBytes
    params: dict[str, Any] = Field(default_factory=dict)


class CharacterConfidence(BaseModel):
    char: str = Field(min_length=1, max_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class TranscribeRunResponse(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    character_confidences: list[CharacterConfidence]

    @field_validator("character_confidences")
    @classmethod
    def align_with_text(
        cls,
        value: list[CharacterConfidence],
        info: Any,
    ) -> list[CharacterConfidence]:
        text = info.data.get("text")
        if text is None:
            return value
        if len(value) != len(text):
            raise ValueError("character_confidences length must match text length")
        for index, entry in enumerate(value):
            if entry.char != text[index]:
                raise ValueError(f"character_confidences[{index}].char must match text[{index}]")
        return value
