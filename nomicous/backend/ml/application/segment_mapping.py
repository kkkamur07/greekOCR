"""Map inference segment output onto the platform's canonical segment DTOs.

This used to hang off the HTTP client that submitted jobs to the inference
service. That client is gone (ADR 0003), but the mapping is not about transport
at all: it is how a wire ``SegmentRunResponse`` — from a job callback or from a
locally executed run — becomes something the merge services can apply.
"""

from __future__ import annotations

from inference.contracts.segment import SegmentRunResponse

from backend.document.infrastructure.orm_models import LineGeometryKind
from backend.ml.domain.segment import CanonicalBlock, CanonicalLine, CanonicalSegmentResult


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
