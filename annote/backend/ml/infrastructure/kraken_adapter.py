"""Kraken segment adapter.

The real Kraken binding is owned by issue 005. Until that catalog/registry is
available, this adapter returns a deterministic canonical result that exercises
the job and merge path without importing FastAPI or GPU-only dependencies.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from backend.ml.domain.segment import (
    CanonicalBlock,
    CanonicalLine,
    CanonicalSegmentResult,
)


class KrakenSegmentAdapter:
    def segment_part(self, image_path: Path | None = None) -> CanonicalSegmentResult:
        if image_path is None:
            raise ValueError("image_path is required for segmentation")
        with Image.open(image_path) as image:
            width, height = image.size

        line_height = max(float(height) * 0.4, 1.0)
        block_height = max(float(height), line_height)
        points = [
            [0.0, 0.0],
            [float(width), 0.0],
            [float(width), line_height],
            [0.0, line_height],
        ]
        return CanonicalSegmentResult(
            blocks=[
                CanonicalBlock(
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
                CanonicalLine(
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
