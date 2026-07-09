"""Layout blocks and lines CRUD for document parts."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.line_geometry import resolve_line_baseline_and_mask
from backend.core.exceptions import NotFoundError, ValidationError
from backend.document.infrastructure.orm_models import Block, Line, LineGeometryKind, LineSource
from backend.document.application.document_service_shared import (
    BLOCK_PATCH_FIELDS,
    LINE_PATCH_FIELDS,
    MAX_REPLACE_PART_LINES,
    DocumentServiceSharedMixin,
)
from backend.users.infrastructure.orm_models import User


class LayoutServiceMixin(DocumentServiceSharedMixin):
    async def list_part_lines(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> list[Line]:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        return await self._documents.list_part_lines(session, part.id)

    async def list_part_layout(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> tuple[list[Block], list[Line]]:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        blocks = await self._list_part_blocks(session, part.id)
        lines = await self._documents.list_part_lines(session, part.id)
        return blocks, lines

    async def create_part_block(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        order: int,
        box: dict,
    ) -> Block:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        block = Block(part_id=part.id, order=order, box=box, manual_geometry=True)
        session.add(block)
        await session.commit()
        await session.refresh(block)
        return block

    async def patch_part_block(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        block_id: UUID,
        **updates: object,
    ) -> Block:
        self._reject_unknown_fields(updates, BLOCK_PATCH_FIELDS, "block patch")
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        block = await self._block_or_404(session, part.id, block_id)
        for key, value in updates.items():
            if value is not None:
                setattr(block, key, value)
        block.manual_geometry = True
        await session.commit()
        await session.refresh(block)
        return block

    async def delete_part_block(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        block_id: UUID,
    ) -> None:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        block = await self._block_or_404(session, part.id, block_id)
        await session.delete(block)
        await session.commit()

    async def create_part_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        order: int,
        kind: LineGeometryKind,
        points: list[list[float]],
        block_id: UUID | None = None,
        baseline: dict | None = None,
        mask: dict | None = None,
    ) -> Line:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        if block_id is not None:
            await self._block_or_404(session, part.id, block_id)
        line = Line(
            part_id=part.id,
            block_id=block_id,
            order=order,
            kind=kind,
            points=points,
            baseline=baseline or {"points": points},
            mask=mask or {"points": points},
            source=LineSource.manual,
            manual_geometry=True,
        )
        session.add(line)
        await session.commit()
        return await self._line_or_404(session, part.id, line.id)

    async def patch_part_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        line_id: UUID,
        **updates: object,
    ) -> Line:
        self._reject_unknown_fields(updates, LINE_PATCH_FIELDS, "line patch")
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        line = await self._line_or_404(session, part.id, line_id)
        if "block_id" in updates and updates["block_id"] is not None:
            await self._block_or_404(session, part.id, updates["block_id"])
        for key, value in updates.items():
            if value is not None:
                setattr(line, key, value)
        line.manual_geometry = True
        line.source = LineSource.manual
        await session.commit()
        return await self._line_or_404(session, part.id, line.id)

    async def delete_part_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        line_id: UUID,
    ) -> None:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        line = await self._line_or_404(session, part.id, line_id)
        await session.delete(line)
        await session.commit()

    async def reset_part_layout(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        line_ids: list[UUID] | None = None,
    ) -> tuple[list[Block], list[Line]]:
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        lines = await self._documents.list_part_lines(session, part.id)
        selected_ids = set(line_ids) if line_ids is not None else {line.id for line in lines}
        if line_ids is not None and selected_ids - {line.id for line in lines}:
            raise NotFoundError("Line not found")
        for line in lines:
            if line.id in selected_ids:
                line.manual_geometry = False
        if line_ids is None:
            blocks = await self._list_part_blocks(session, part.id)
            for block in blocks:
                block.manual_geometry = False
        await session.commit()
        return await self.list_part_layout(session, user, project_id, document_id, part_id)

    async def replace_part_lines(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        lines: list[dict],
        allow_new_ids: bool = False,
    ) -> list[Line]:
        if len(lines) > MAX_REPLACE_PART_LINES:
            raise ValidationError(
                f"Cannot replace more than {MAX_REPLACE_PART_LINES} lines at once"
            )
        document = await self.get_document(session, user, project_id, document_id)
        part = await self._document_part_or_404(session, document, part_id)
        ground_truth = await self._ensure_ground_truth_transcription(session, document)

        requested_ids = [line["id"] for line in lines if line.get("id") is not None]
        if len(set(requested_ids)) != len(requested_ids):
            raise ValidationError("Line ids must be unique")

        existing = await self._documents.list_part_lines(session, part.id)
        existing_by_id = {line.id: line for line in existing}
        if not allow_new_ids:
            new_supplied_ids = [
                line_id for line_id in requested_ids if line_id not in existing_by_id
            ]
            if new_supplied_ids:
                raise ValidationError("New line ids are server-generated")
        requested_id_set = set(requested_ids)
        for line in existing:
            if line.id not in requested_id_set:
                await session.delete(line)

        for data in lines:
            line_id = data.get("id")
            prior = existing_by_id.get(line_id) if line_id is not None else None
            line = prior
            if line is None:
                line = Line(part_id=part.id, baseline={}, mask=None)
                if line_id is not None:
                    line.id = line_id
                session.add(line)

            points = data["points"]
            if "block_id" in data:
                block_id = data["block_id"]
                if block_id is not None:
                    await self._block_or_404(session, part.id, block_id)
                line.block_id = block_id
            elif prior is not None:
                line.block_id = prior.block_id
            line.order = data["order"]
            line.kind = data["kind"]
            line.points = points
            line.baseline, line.mask = resolve_line_baseline_and_mask(
                points=points,
                payload_baseline=data.get("baseline"),
                payload_mask=data.get("mask"),
                existing_baseline=prior.baseline if prior is not None else None,
                existing_mask=prior.mask if prior is not None else None,
            )
            line.source = data["source"]
            if "source_metadata" in data:
                line.source_metadata = data["source_metadata"]
            elif prior is not None:
                line.source_metadata = prior.source_metadata
            if "kraken_ceiling" in data:
                line.kraken_ceiling = data["kraken_ceiling"]
            elif prior is not None:
                line.kraken_ceiling = prior.kraken_ceiling
            source_value = (
                data["source"].value if hasattr(data["source"], "value") else data["source"]
            )
            line.manual_geometry = source_value == "manual"

            await self._set_ground_truth_text(
                line, ground_truth, data.get("approved_text"), session
            )

        await session.commit()
        return await self._documents.list_part_lines(session, part.id)
