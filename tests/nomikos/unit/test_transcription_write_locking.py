"""Every writer that can make a line untouchable holds the part while it does.

Segment health decides a deletion from a snapshot of which lines carry human
work, and `page_transcription_lines.paired_line_id` is ON DELETE SET NULL. A
pairing that commits inside that snapshot's window is therefore removed by the
delete with no error raised and nothing recording that it existed. The part lock
is what stops the two overlapping, and it only works if every such writer takes
it, so what is asserted here is simply that each one asks for it.

The lock is invisible to a single caller by design: nothing about the result of
a pairing changes when it is dropped. That is exactly why it needs a test of its
own rather than being covered incidentally.
"""

from __future__ import annotations

import uuid

import pytest

from backend.document.application.document_access import PartContext
from backend.document.application.transcription_service import TranscriptionService
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    Line,
    PageTranscriptionLine,
    Transcription,
    TranscriptionKind,
)


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

    async def require_document(self, *_args, **_kwargs):
        return PartContext(project=object(), document=self._document, part=self._part)


class _StubGroundTruth:
    def __init__(self, layer: Transcription) -> None:
        self._layer = layer

    async def layer_for(self, _session, _document) -> Transcription:
        return self._layer

    async def write(self, _session, _line, _layer, _text) -> None:
        pass


class _StubRepository:
    def __init__(self, line: Line, text_line: PageTranscriptionLine) -> None:
        self._line = line
        self._text_line = text_line
        self.locks: list[uuid.UUID] = []

    async def lock_part(self, _session, part_id) -> None:
        self.locks.append(part_id)

    async def lock_parts(self, _session, part_ids) -> None:
        self.locks.extend(sorted(set(part_ids)))

    async def get_line_in_part(self, _session, _part_id, _line_id):
        return self._line

    async def get_page_transcription_line(self, _session, _part_id, _order):
        return self._text_line

    async def list_page_transcription_lines(self, _session, _part_id):
        return [self._text_line]

    async def count_part_lines(self, _session, _part_id) -> int:
        return 1

    async def count_paired_ground_truth_lines(self, _session, _part_id) -> int:
        return 1


def _service() -> tuple[TranscriptionService, _StubRepository, DocumentPart, Line]:
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="codex")
    part = DocumentPart(id=uuid.uuid4(), document_id=document.id, order=0, image_key="page.webp")
    line = Line(id=uuid.uuid4(), part_id=part.id, order=0, baseline={}, points=[])
    text_line = PageTranscriptionLine(
        id=uuid.uuid4(), part_id=part.id, order=0, text="alpha", paired_line_id=None
    )
    layer = Transcription(
        id=uuid.uuid4(),
        document_id=document.id,
        name="Ground truth",
        kind=TranscriptionKind.ground_truth,
    )
    repository = _StubRepository(line, text_line)
    service = TranscriptionService(
        documents=repository,
        access=_StubAccess(document, part),
        ground_truth=_StubGroundTruth(layer),
    )
    return service, repository, part, line


@pytest.mark.asyncio
async def test_pairing_a_text_line_locks_the_part() -> None:
    """The review's case: pair a segment while a suspect deletion is in flight."""
    service, repository, part, line = _service()

    await service.pair_page_text_line(
        _Session(),
        object(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        line_id=line.id,
        text_line_order=0,
    )

    assert repository.locks == [part.id]


@pytest.mark.asyncio
async def test_a_ground_truth_text_edit_locks_the_line_s_part() -> None:
    """The other way a line becomes untouchable, on a route addressed by document."""
    service, repository, part, line = _service()
    layer = await service._ground_truth.layer_for(None, None)

    async def _transcription_or_404(*_args, **_kwargs):
        return layer

    async def _line_in_document_or_404(*_args, **_kwargs):
        return line

    service._transcription_or_404 = _transcription_or_404  # type: ignore[method-assign]
    service._line_in_document_or_404 = _line_in_document_or_404  # type: ignore[method-assign]

    class _TextSession(_Session):
        async def execute(self, _stmt):
            class _Result:
                @staticmethod
                def scalar_one_or_none():
                    return None

            return _Result()

        def add(self, _item: object) -> None:
            pass

    await service.patch_ground_truth_line_text(
        _TextSession(),
        object(),
        uuid.uuid4(),
        part.document_id,
        layer.id,
        line.id,
        text="ܡܪܝܐ",
    )

    assert repository.locks == [line.part_id]
