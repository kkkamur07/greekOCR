"""PATCH bodies must distinguish an omitted field from an explicit null.

The line/block patch handlers dropped every ``None`` value, so a client could never clear
``lines.block_id`` (a nullable column) — the update was silently ignored and the response
still carried the old block. Fields backing NOT NULL columns reject an explicit null the
same way ``DocumentUpdateRequest`` already did.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError as PydanticValidationError

from backend.document.api.schemas import BlockPatchRequest, LinePatchRequest
from backend.document.application.document_access import PartContext
from backend.document.application.layout_service import LayoutService
from backend.document.infrastructure.orm_models import Block, Document, DocumentPart, Line


class _Session:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, _item: object) -> None:
        pass


class _StubAccess:
    def __init__(self, document: Document, part: DocumentPart) -> None:
        self._document = document
        self._part = part

    async def require_part(self, *_args, **_kwargs) -> PartContext:
        return PartContext(project=object(), document=self._document, part=self._part)


class _StubRepository:
    """Only the two row lookups ``patch_part_line`` performs."""

    def __init__(self, line: Line | None, block: Block | None) -> None:
        self._line = line
        self._block = block

    async def get_line_in_part(self, _session, _part_id, _line_id):
        return self._line

    async def get_block_in_part(self, _session, _part_id, _block_id):
        return self._block


def _service_with(monkeypatch, *, part: DocumentPart, line: Line | None, block: Block | None):
    document = Document(id=part.document_id, name="codex")
    service = LayoutService(
        documents=_StubRepository(line, block),
        access=_StubAccess(document, part),
    )
    return service, document


# --- Request schema: omitted vs explicit null ---


def test_line_patch_separates_omitted_from_explicit_null() -> None:
    assert LinePatchRequest().model_dump(exclude_unset=True) == {}
    assert LinePatchRequest.model_validate({"block_id": None}).model_dump(exclude_unset=True) == {
        "block_id": None
    }
    assert LinePatchRequest.model_validate({"mask": None}).model_dump(exclude_unset=True) == {
        "mask": None
    }


@pytest.mark.parametrize("field", ["order", "baseline", "points"])
def test_line_patch_rejects_null_for_not_null_columns(field: str) -> None:
    with pytest.raises(PydanticValidationError):
        LinePatchRequest.model_validate({field: None})


@pytest.mark.parametrize("field", ["order", "box"])
def test_block_patch_rejects_null_for_not_null_columns(field: str) -> None:
    with pytest.raises(PydanticValidationError):
        BlockPatchRequest.model_validate({field: None})


# --- Handler: an explicit null actually clears the column ---


@pytest.mark.asyncio
async def test_patch_line_clears_block_id_when_explicitly_null(monkeypatch) -> None:
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="k")
    line = Line(id=uuid.uuid4(), part_id=part.id, block_id=uuid.uuid4(), baseline={}, order=0)
    service, _ = _service_with(monkeypatch, part=part, line=line, block=None)
    body = LinePatchRequest.model_validate({"block_id": None})

    patched = await service.patch_part_line(
        _Session(),
        object(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        line.id,
        **body.model_dump(exclude_unset=True),
    )

    assert patched.block_id is None


@pytest.mark.asyncio
async def test_patch_line_leaves_block_id_alone_when_omitted(monkeypatch) -> None:
    block_id = uuid.uuid4()
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="k")
    line = Line(id=uuid.uuid4(), part_id=part.id, block_id=block_id, baseline={}, order=0)
    service, _ = _service_with(monkeypatch, part=part, line=line, block=None)
    body = LinePatchRequest.model_validate({"order": 4})

    patched = await service.patch_part_line(
        _Session(),
        object(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        line.id,
        **body.model_dump(exclude_unset=True),
    )

    assert patched.block_id == block_id
    assert patched.order == 4


@pytest.mark.asyncio
async def test_patch_line_clears_mask_when_explicitly_null(monkeypatch) -> None:
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="k")
    line = Line(
        id=uuid.uuid4(),
        part_id=part.id,
        baseline={"points": [[0, 0], [1, 1]]},
        mask={"points": [[0, 0], [1, 1]]},
        order=0,
    )
    service, _ = _service_with(monkeypatch, part=part, line=line, block=None)
    body = LinePatchRequest.model_validate({"mask": None})

    patched = await service.patch_part_line(
        _Session(),
        object(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        line.id,
        **body.model_dump(exclude_unset=True),
    )

    assert patched.mask is None
