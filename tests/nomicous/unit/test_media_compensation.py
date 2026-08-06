"""Unit coverage for the upload path's object-store calls.

Two things are held here. Compensation: a database commit that fails after the bytes
landed must queue a deletion intent rather than orphan them. And offload: every store
call on that path is a synchronous HTTPS round trip, so it has to run on a worker
thread - a 100 MiB upload executed on the loop thread stalls every other request for
the duration of the transfer.
"""

from __future__ import annotations

import threading
import uuid

import pytest

from backend.document.application.document_access import DocumentContext
from backend.document.application.part_service import DocumentPartService
from backend.document.infrastructure.media_store.encoding import DecodedPartImage
from backend.document.infrastructure.orm_models import Document


class _CommitFailingSession:
    def __init__(self) -> None:
        self.rollbacks = 0

    def add(self, _item) -> None:
        pass

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        raise RuntimeError("database commit unavailable")

    async def rollback(self) -> None:
        self.rollbacks += 1


class _CompensatingRepository:
    def __init__(self) -> None:
        self.intent_keys: list[str] = []

    async def next_part_order(self, _session, _document_id) -> int:
        return 0

    async def enqueue_media_deletion_intent(self, _session, image_key: str) -> None:
        self.intent_keys.append(image_key)


class _RecordingStore:
    """Records the thread each call ran on, so the tests can see the offload."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.deletes: list[str] = []
        self.write_threads: list[int] = []
        self.delete_threads: list[int] = []

    def part_image_key(self, _part_id, **_kwargs) -> str:
        return "parts/compensation.webp"

    def write(self, image_key: str, _data: bytes) -> None:
        self.writes.append(image_key)
        self.write_threads.append(threading.get_ident())

    def delete(self, image_key: str) -> None:
        self.deletes.append(image_key)
        self.delete_threads.append(threading.get_ident())


class _DeleteFailingStore(_RecordingStore):
    def delete(self, image_key: str) -> None:
        super().delete(image_key)
        raise RuntimeError("object storage unavailable")


class _StubAccess:
    """The upload is already authorized by the time compensation matters."""

    def __init__(self, document: Document) -> None:
        self._document = document

    async def require_document(self, *_args, **_kwargs) -> DocumentContext:
        return DocumentContext(project=object(), document=self._document)


@pytest.mark.asyncio
async def test_failed_upload_commit_records_compensating_delete_intent(monkeypatch) -> None:
    repo = _CompensatingRepository()
    store = _DeleteFailingStore()
    document = Document(id=uuid.uuid4(), name="test")
    service = DocumentPartService(documents=repo, media=store, access=_StubAccess(document))

    monkeypatch.setattr(
        "backend.document.application.part_service.encode_part_image_with_size",
        lambda _data: DecodedPartImage(data=b"encoded", width=4, height=6),
    )
    session = _CommitFailingSession()

    with pytest.raises(RuntimeError, match="database commit unavailable"):
        await service.upload_part(
            session,
            user=object(),
            project_id=uuid.uuid4(),
            document_id=document.id,
            data=b"source",
        )

    assert store.writes == ["parts/compensation.webp"]
    assert store.deletes == ["parts/compensation.webp"]
    assert repo.intent_keys == ["parts/compensation.webp"]
    assert session.rollbacks == 1


# --- Offload: neither store call may run on the event loop thread ---


class _CommittingSession:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def add(self, _item) -> None:
        pass

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def refresh(self, _item) -> None:
        pass


def _upload_service(store, monkeypatch) -> tuple[DocumentPartService, Document]:
    document = Document(id=uuid.uuid4(), name="test")
    service = DocumentPartService(
        documents=_CompensatingRepository(), media=store, access=_StubAccess(document)
    )
    monkeypatch.setattr(
        "backend.document.application.part_service.encode_part_image_with_size",
        lambda _data: DecodedPartImage(data=b"encoded", width=4, height=6),
    )
    return service, document


@pytest.mark.asyncio
async def test_upload_writes_the_blob_off_the_event_loop(monkeypatch) -> None:
    store = _RecordingStore()
    service, document = _upload_service(store, monkeypatch)

    await service.upload_part(
        _CommittingSession(),
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        data=b"source",
    )

    assert store.writes == ["parts/compensation.webp"]
    assert threading.get_ident() not in store.write_threads
