"""Segment merge - replace the machine geometry nobody has touched, keep everything a human did.

A segment job redraws a page. What it must never do is throw away work: ``manual_geometry``
records that someone moved a polygon, but that is only one of the ways a human touches a
line. Approving its text writes a ground-truth transcription; pairing it with a line of
imported page text writes a ``page_transcription_lines`` row. Neither sets
``manual_geometry``, so a page where forty lines were corrected and approved without a
single polygon being nudged looks, to a geometry-only guard, exactly like a page nobody
has opened. The guard here is the union of the three signals: a line survives a re-segment
if it has manual geometry, **or** approved text, **or** a pairing.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.core.exceptions import NotFoundError
from backend.document.infrastructure.orm_models import (
    Block,
    DocumentPart,
    Line,
    LineSource,
    LineTranscription,
    PageTranscriptionLine,
    Transcription,
    TranscriptionKind,
)
from backend.ml.domain.segment import CanonicalSegmentResult


@dataclass(frozen=True)
class SegmentMergeSummary:
    blocks_count: int
    lines_count: int
    added_lines: int
    pruned_lines: int
    preserved_manual_lines: int
    # Lines whose geometry the machine drew but which a human has since approved text
    # on or paired. Reported separately so a job result still says how many polygons
    # on the page are hand-drawn.
    preserved_transcribed_lines: int = 0


class SegmentMergeService:
    """Apply canonical segment output to a part without touching what a human did."""

    def apply_sync(
        self,
        session: Session,
        *,
        part_id: UUID,
        canonical_segment: CanonicalSegmentResult,
        job_id: UUID,
        commit: bool = True,
    ) -> SegmentMergeSummary:
        part = self._load_part(session, part_id)
        transcribed_ids = self._human_transcribed_line_ids(session, part)

        preserved_manual_lines = 0
        preserved_transcribed_lines = 0
        pruned_lines = 0
        for line in list(part.lines):
            if line.manual_geometry:
                preserved_manual_lines += 1
                continue
            if line.id in transcribed_ids:
                preserved_transcribed_lines += 1
                continue
            session.delete(line)
            pruned_lines += 1
        # Blocks carry no transcription of their own, so the geometry rule is the whole
        # rule for them. A kept line whose block goes loses only its block_id (SET NULL),
        # which is the state kraken lines without a block are already in.
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

        if commit:
            session.commit()
        return SegmentMergeSummary(
            blocks_count=len(canonical_segment.blocks),
            lines_count=len(canonical_segment.lines),
            added_lines=len(canonical_segment.lines),
            pruned_lines=pruned_lines,
            preserved_manual_lines=preserved_manual_lines,
            preserved_transcribed_lines=preserved_transcribed_lines,
        )

    def _human_transcribed_line_ids(self, session: Session, part: DocumentPart) -> set[UUID]:
        """Lines on ``part`` that carry approved text or a pairing.

        Only the ground-truth layer counts: a model's own output on a line is what a
        re-segment is allowed to replace. A pairing counts on its own because pairing
        writes the ground-truth row and un-pairing removes it, so the two normally agree,
        and when they do not (imported text paired before any approval, a ground-truth
        layer created later) the pairing is still a human's decision about that line.
        """
        line_ids = [line.id for line in part.lines]
        if not line_ids:
            return set()
        transcribed = session.scalars(
            select(LineTranscription.line_id)
            .join(Transcription, Transcription.id == LineTranscription.transcription_id)
            .where(
                LineTranscription.line_id.in_(line_ids),
                Transcription.kind == TranscriptionKind.ground_truth,
            )
        )
        paired = session.scalars(
            select(PageTranscriptionLine.paired_line_id).where(
                PageTranscriptionLine.part_id == part.id,
                PageTranscriptionLine.paired_line_id.is_not(None),
            )
        )
        return set(transcribed) | set(paired)

    def _load_part(self, session: Session, part_id: UUID) -> DocumentPart:
        # Lock the part row for the length of the merge. Two segment jobs on the
        # same part each delete-then-reinsert its machine geometry; the callback's
        # FOR UPDATE covers only the Job row, not the part, so without this lock
        # the two merges interleave their delete/insert and leave a corrupted mix
        # of both runs' blocks and lines. The second job blocks here until the
        # first commits, then reads the fresh state and cleanly replaces it.
        result = session.execute(
            select(DocumentPart)
            .where(DocumentPart.id == part_id)
            .with_for_update()
            .options(selectinload(DocumentPart.blocks))
            .options(selectinload(DocumentPart.lines))
        )
        part = result.scalar_one_or_none()
        if part is None:
            raise NotFoundError("Part not found")
        return part
