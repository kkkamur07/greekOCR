"""Transcribe task request/response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from inference.contracts.common import (
    MAX_GEOMETRY_POINTS,
    MAX_LINE_TEXT_CHARS,
    MAX_TRANSCRIBE_LINES,
)


class CharacterConfidence(BaseModel):
    char: str = Field(min_length=1, max_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class TranscribeRunResponse(BaseModel):
    text: str = Field(max_length=MAX_LINE_TEXT_CHARS)
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
    points: list[list[float]] | None = Field(default=None, max_length=MAX_GEOMETRY_POINTS)


# Client-visible text for a line the runtime could not transcribe. Static on
# purpose: this response travels to the platform as a job callback and on to a
# researcher's browser, and the real exception belongs in the log of whichever
# machine ran the page, never in a field somebody reads as transcribed text.
TRANSCRIBE_LINE_ERROR = "Line could not be transcribed"


class TranscribeBatchLineResult(BaseModel):
    line_id: str | None = None
    line_index: int = Field(ge=0)
    # Exactly one of ``output``/``error`` is set. ``error`` is absent on every
    # successful line, so a consumer that only reads ``output`` still gets it
    # on every line that succeeded; it only needs to handle the case where one
    # line of a page failed and the rest did not.
    output: TranscribeRunResponse | None = None
    error: str | None = None

    @model_validator(mode="after")
    def require_output_or_error(self) -> TranscribeBatchLineResult:
        if (self.output is None) == (self.error is None):
            raise ValueError("line result must carry exactly one of output or error")
        return self


class TranscribeBatchRunResponse(BaseModel):
    lines: list[TranscribeBatchLineResult] = Field(min_length=1, max_length=MAX_TRANSCRIBE_LINES)

    @model_validator(mode="after")
    def require_one_transcribed_line(self) -> TranscribeBatchRunResponse:
        # Partial success is a batch where *something* came back. A batch where
        # every line failed is a failed run and has to be raised as one, or the
        # caller stores an empty transcription and calls it done.
        if all(line.output is None for line in self.lines):
            raise ValueError("transcribe batch must contain at least one transcribed line")
        return self
