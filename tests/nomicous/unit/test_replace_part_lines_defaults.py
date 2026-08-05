"""PUT /lines accepts payloads that omit the optional ``kind``/``source`` fields.

``model_dump(exclude_unset=True)`` strips fields the client never sent, so the service
must read them with the schema's defaults instead of subscripting the dict.
"""

from __future__ import annotations

import uuid

import pytest

from backend.document.api.schemas import LinesReplaceRequest
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    LineGeometryKind,
    LineSource,
    Transcription,
    TranscriptionKind,
)

_POINTS = [[10.0, 10.0], [40.0, 10.0], [40.0, 25.0], [10.0, 25.0]]


class _Session:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.commits = 0

    def add(self, item: object) -> None:
        self.added.append(item)

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        self.commits += 1

    async def delete(self, item: object) -> None:
        self.deleted.append(item)


class _Repository:
    def __init__(self, part: DocumentPart, ground_truth: Transcription) -> None:
        self._part = part
        self._ground_truth = ground_truth
        self.persisted: list[object] = []

    async def get_part(self, _session, part_id):
        return self._part if part_id == self._part.id else None

    async def get_ground_truth_transcription(self, _session, _document_id):
        return self._ground_truth

    async def list_part_lines(self, _session, _part_id):
        # Mirrors the repository contract: the rows the service just wrote back.
        return list(self.persisted)


def _service(monkeypatch) -> tuple[DocumentService, _Session, Document, DocumentPart, _Repository]:
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="codex")
    part = DocumentPart(id=uuid.uuid4(), document_id=document.id, order=0, image_key="page.webp")
    ground_truth = Transcription(
        id=uuid.uuid4(),
        document_id=document.id,
        name="Ground truth",
        kind=TranscriptionKind.ground_truth,
    )
    repository = _Repository(part, ground_truth)
    service = DocumentService(documents=repository)

    async def get_document(*_args, **_kwargs):
        return document

    monkeypatch.setattr(service, "get_document", get_document)
    session = _Session()

    original_add = session.add

    def add(item: object) -> None:
        original_add(item)
        repository.persisted.append(item)

    session.add = add  # type: ignore[method-assign]
    return service, session, document, part, repository


def _payload(**overrides: object) -> list[dict]:
    body = LinesReplaceRequest.model_validate(
        {"lines": [{"order": 0, "points": _POINTS, **overrides}]}
    )
    return [line.model_dump(exclude_unset=True) for line in body.lines]


@pytest.mark.asyncio
async def test_replace_part_lines_defaults_kind_and_source_when_omitted(monkeypatch) -> None:
    service, session, document, part, _repository = _service(monkeypatch)
    payload = _payload()
    # The regression is only reachable when the client omits both optional fields.
    assert "kind" not in payload[0]
    assert "source" not in payload[0]

    lines = await service.replace_part_lines(
        session,
        user=object(),
        project_id=document.project_id,
        document_id=document.id,
        part_id=part.id,
        lines=payload,
    )

    assert len(lines) == 1
    assert lines[0].kind is LineGeometryKind.polygon
    assert lines[0].source is LineSource.manual
    assert lines[0].manual_geometry is True
    assert lines[0].points == _POINTS
    assert session.commits == 1


@pytest.mark.asyncio
async def test_replace_part_lines_still_honours_explicit_kind_and_source(monkeypatch) -> None:
    service, session, document, part, _repository = _service(monkeypatch)

    lines = await service.replace_part_lines(
        session,
        user=object(),
        project_id=document.project_id,
        document_id=document.id,
        part_id=part.id,
        lines=_payload(kind="polygon", source="kraken"),
    )

    assert lines[0].source is LineSource.kraken
    assert lines[0].manual_geometry is False
