"""Map inference segment output onto the platform's canonical segment DTOs.

Converts a wire ``SegmentRunResponse`` (from a job callback or a locally
executed run) into the form the merge services apply.
"""

from __future__ import annotations

from backend.document.infrastructure.orm_models import LineGeometryKind
from backend.ml.domain.segment import CanonicalBlock, CanonicalLine, CanonicalSegmentResult
from nomikos_inference.contracts.segment import SegmentRunResponse


def to_canonical_segment(output: SegmentRunResponse) -> CanonicalSegmentResult:
    return CanonicalSegmentResult(
        blocks=[
            CanonicalBlock(
                external_id=block.external_id,
                order=block.order,
                box=block.box,
            )
            for block in output.blocks
        ],
        lines=[
            CanonicalLine(
                external_id=line.external_id,
                order=line.order,
                block_external_id=line.block_external_id,
                baseline=line.baseline,
                mask=line.mask,
                kind=LineGeometryKind(line.kind.value),
                points=line.points,
                kraken_ceiling=line.kraken_ceiling,
                source_metadata=line.source_metadata,
            )
            for line in output.lines
        ],
    )
