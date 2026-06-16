"""Segment merge — preserve manual geometry, replace machine geometry."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.core.exceptions import NotFoundError
from backend.document.infrastructure.orm_models import Block, DocumentPart, Line, LineSource
from backend.inference.domain.segment import CanonicalSegmentResult


@dataclass(frozen=True)
class SegmentMergeSummary:
    blocks_count: int
    lines_count: int
    added_lines: int
    pruned_lines: int
    preserved_manual_lines: int


class SegmentMergeService:
    """Apply canonical segment output to a part without touching manual geometry."""

    def apply_sync(
        self,
        session: Session,
        *,
        part_id: UUID,
        canonical_segment: CanonicalSegmentResult,
        job_id: UUID,
    ) -> SegmentMergeSummary:
        part = self._load_part(session, part_id)

        preserved_manual_lines = sum(1 for line in part.lines if line.manual_geometry)
        pruned_lines = sum(1 for line in part.lines if not line.manual_geometry)

        for line in list(part.lines):
            if not line.manual_geometry:
                session.delete(line)
        for block in list(part.blocks):
            if not block.manual_geometry:
                session.delete(block)
        session.flush()

        blocks_by_external_id: dict[str, Block] = {}
        for block_data in canonical_segment.blocks:
            block = Block(
                part_id=part.id,
                order=block_data.order,
                box=block_data.box,
                manual_geometry=False,
            )
            session.add(block)
            blocks_by_external_id[block_data.external_id] = block
        session.flush()

        for line_data in canonical_segment.lines:
            source_metadata = {
                **line_data.source_metadata,
                "external_id": line_data.external_id,
                "job_id": str(job_id),
            }
            line = Line(
                part_id=part.id,
                block_id=(
                    blocks_by_external_id[line_data.block_external_id].id
                    if line_data.block_external_id in blocks_by_external_id
                    else None
                ),
                baseline=line_data.baseline,
                mask=line_data.mask,
                kind=line_data.kind,
                points=line_data.points,
                source=LineSource.kraken,
                source_metadata=source_metadata,
                kraken_ceiling=line_data.kraken_ceiling,
                manual_geometry=False,
                order=line_data.order,
            )
            session.add(line)

        session.commit()
        return SegmentMergeSummary(
            blocks_count=len(canonical_segment.blocks),
            lines_count=len(canonical_segment.lines),
            added_lines=len(canonical_segment.lines),
            pruned_lines=pruned_lines,
            preserved_manual_lines=preserved_manual_lines,
        )

    def _load_part(self, session: Session, part_id: UUID) -> DocumentPart:
        result = session.execute(
            select(DocumentPart)
            .options(selectinload(DocumentPart.blocks))
            .options(selectinload(DocumentPart.lines))
            .where(DocumentPart.id == part_id)
        )
        part = result.scalar_one_or_none()
        if part is None:
            raise NotFoundError("Part not found")
        return part
