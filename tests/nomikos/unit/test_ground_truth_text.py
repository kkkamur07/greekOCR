"""``GroundTruthText`` — the one piece of behaviour three modules share.

Layout's bulk replace, page-transcription pairing, and the ground-truth layer edits all
write through here, so the invariants are asserted once, at the interface, rather than
three times through whichever endpoint happens to reach them.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import pytest

from backend.document.application.ground_truth import GroundTruthText
from backend.document.infrastructure.orm_models import (
    Document,
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)


class _Session:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.flushes = 0

    def add(self, item: object) -> None:
        self.added.append(item)

    @asynccontextmanager
    async def begin_nested(self):
        # SAVEPOINT stand-in: yield, and let an exception in the block propagate
        # the way ``layer_for`` expects (it catches IntegrityError around this).
        yield

    async def flush(self) -> None:
        self.flushes += 1

    async def delete(self, item: object) -> None:
        self.deleted.append(item)


class _Repository:
    def __init__(self, existing: Transcription | None = None) -> None:
        self._existing = existing
        self.lookups = 0

    async def get_ground_truth_transcription(self, _session, _document_id):
        self.lookups += 1
        return self._existing


def _layer() -> Transcription:
    return Transcription(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        name="Ground truth",
        kind=TranscriptionKind.ground_truth,
    )


def _line(*rows: LineTranscription) -> Line:
    line = Line(id=uuid.uuid4(), part_id=uuid.uuid4(), order=0, baseline={})
    line.transcriptions = list(rows)
    return line


# --- One layer per document, created on demand ---


@pytest.mark.asyncio
async def test_existing_layer_is_reused_and_nothing_is_written() -> None:
    existing = _layer()
    session = _Session()
    writer = GroundTruthText(documents=_Repository(existing))

    assert await writer.layer_for(session, Document(id=uuid.uuid4(), name="c")) is existing
    assert session.added == []


@pytest.mark.asyncio
async def test_missing_layer_is_created_and_flushed_not_committed() -> None:
    """Flush, not commit: the caller is mid-transaction and owns the commit."""
    session = _Session()
    writer = GroundTruthText(documents=_Repository(None))
    document = Document(id=uuid.uuid4(), name="c")

    layer = await writer.layer_for(session, document)

    assert layer.kind is TranscriptionKind.ground_truth
    assert layer.document_id == document.id
    assert session.added == [layer]
    assert session.flushes == 1


# --- Writing text onto a line ---


@pytest.mark.asyncio
async def test_first_write_appends_a_row_with_no_confidence() -> None:
    layer = _layer()
    line = _line()
    writer = GroundTruthText(documents=_Repository(layer))

    await writer.write(_Session(), line, layer, "ἐν ἀρχῇ")

    assert len(line.transcriptions) == 1
    assert line.transcriptions[0].text == "ἐν ἀρχῇ"
    assert line.transcriptions[0].confidence is None


@pytest.mark.asyncio
async def test_rewrite_updates_in_place_and_drops_any_model_confidence() -> None:
    """A confidence score describes a model's guess; once a human types, it describes nothing."""
    layer = _layer()
    row = LineTranscription(transcription_id=layer.id, text="old", confidence=0.42)
    line = _line(row)
    writer = GroundTruthText(documents=_Repository(layer))

    await writer.write(_Session(), line, layer, "new")

    assert len(line.transcriptions) == 1
    assert (row.text, row.confidence) == ("new", None)


@pytest.mark.asyncio
async def test_none_removes_the_row_rather_than_storing_an_empty_string() -> None:
    layer = _layer()
    row = LineTranscription(transcription_id=layer.id, text="paired", confidence=None)
    line = _line(row)
    session = _Session()
    writer = GroundTruthText(documents=_Repository(layer))

    await writer.write(session, line, layer, None)

    assert line.transcriptions == []
    assert session.deleted == [row]


@pytest.mark.asyncio
async def test_none_on_a_line_that_has_no_ground_truth_is_a_no_op() -> None:
    layer = _layer()
    line = _line()
    session = _Session()
    writer = GroundTruthText(documents=_Repository(layer))

    await writer.write(session, line, layer, None)

    assert session.deleted == []


@pytest.mark.asyncio
async def test_other_layers_on_the_same_line_are_untouched() -> None:
    layer = _layer()
    model_row = LineTranscription(transcription_id=uuid.uuid4(), text="model", confidence=0.9)
    line = _line(model_row)
    writer = GroundTruthText(documents=_Repository(layer))

    await writer.write(_Session(), line, layer, "human")

    assert model_row.text == "model"
    assert model_row.confidence == 0.9
    assert len(line.transcriptions) == 2
