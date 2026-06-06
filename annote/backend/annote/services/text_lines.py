"""Text line parser — split page transcription on line breaks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextLine:
    index: int
    text: str


def split_text_lines(text: str) -> list[TextLine]:
    """Split transcription text into numbered text lines (1-based index)."""
    if not text or not text.strip():
        return []
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return [TextLine(index=i + 1, text=line) for i, line in enumerate(lines)]
