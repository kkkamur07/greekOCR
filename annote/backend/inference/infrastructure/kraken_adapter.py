"""Kraken segment adapter.

The real Kraken binding is owned by issue 005. Until that catalog/registry is
available, this adapter returns a deterministic canonical result that exercises
the job and merge path without importing FastAPI or GPU-only dependencies.
"""

from __future__ import annotations

from pathlib import Path

from backend.inference.domain.segment import (
    CanonicalBlock,
    CanonicalLine,
    CanonicalSegmentResult,
)


class KrakenSegmentAdapter:
    def segment_part(self, _image_path: Path | None = None) -> CanonicalSegmentResult:
        points = [[0.0, 0.0], [20.0, 0.0], [20.0, 8.0], [0.0, 8.0]]
        return CanonicalSegmentResult(
            blocks=[
                CanonicalBlock(
                    external_id="kraken-block-1",
                    order=0,
                    box={"points": [[0.0, 0.0], [20.0, 0.0], [20.0, 12.0], [0.0, 12.0]]},
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
