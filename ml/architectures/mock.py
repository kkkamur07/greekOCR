"""Test-only synthetic inference helpers for lightweight contract tests."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from ml.contracts.segment import SegmentBlock, SegmentLine, SegmentRunResponse


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
