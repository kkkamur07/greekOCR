"""Deterministic segment runner for tests and local dev without Kraken weights."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from ml.contracts.segment import SegmentBlock, SegmentLine, SegmentRunResponse
from ml.contracts.transcribe import CharacterConfidence, TranscribeRunResponse


def mock_segment(image_bytes: bytes) -> SegmentRunResponse:
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size

    line_height = max(float(height) * 0.4, 1.0)
    block_height = max(float(height), line_height)
    points = [
        [0.0, 0.0],
        [float(width), 0.0],
        [float(width), line_height],
        [0.0, line_height],
    ]
    return SegmentRunResponse(
        blocks=[
            SegmentBlock(
                external_id="kraken-block-1",
                order=0,
                box={
                    "points": [
                        [0.0, 0.0],
                        [float(width), 0.0],
                        [float(width), block_height],
                        [0.0, block_height],
                    ]
                },
            )
        ],
        lines=[
            SegmentLine(
                external_id="kraken-line-1",
                order=0,
                block_external_id="kraken-block-1",
                baseline={"points": points},
                mask={"points": points},
                points=points,
                kraken_ceiling=points,
                source_metadata={"adapter": "kraken_stub"},
            )
        ],
    )


def mock_transcribe(image_bytes: bytes, *, params: dict) -> TranscribeRunResponse:
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
