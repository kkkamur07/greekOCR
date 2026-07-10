"""Unit coverage for media compensation when database commits fail."""

from __future__ import annotations

import uuid

import pytest

from backend.document.application.document_service import DocumentService
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


class _DeleteFailingStore:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.deletes: list[str] = []

    def part_image_key(self, _part_id, **_kwargs) -> str:
        return "parts/compensation.webp"

    def write(self, image_key: str, _data: bytes) -> None:
        self.writes.append(image_key)

    def delete(self, image_key: str) -> None:
        self.deletes.append(image_key)
        raise RuntimeError("object storage unavailable")


@pytest.mark.asyncio
async def test_failed_upload_commit_records_compensating_delete_intent(monkeypatch) -> None:
    repo = _CompensatingRepository()
    store = _DeleteFailingStore()
    service = DocumentService(documents=repo, media=store)
    document = Document(id=uuid.uuid4(), name="test")

    async def get_document(*_args, **_kwargs):
        return document

    monkeypatch.setattr(service, "get_document", get_document)
    monkeypatch.setattr(
        "backend.document.application.part_service.encode_part_image",
        lambda _data: b"encoded",
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
