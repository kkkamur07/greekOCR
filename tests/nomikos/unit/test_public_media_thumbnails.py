"""The unauthenticated media route only renders a closed set of thumbnail widths."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.document.api import public_media, public_rate_limit
from backend.document.api.media_responses import PUBLIC_THUMBNAIL_WIDTHS
from infrastructure.db import get_db

PART_ID = uuid.uuid4()


class _StubService:
    def __init__(self) -> None:
        self.reads: list[int | None] = []

    async def get_part_for_public_media(self, session, part_id):
        return SimpleNamespace(id=part_id, image_key="page.webp")

    async def read_part_bytes(self, part, *, width=None) -> bytes:
        self.reads.append(width)
        return b"webp-bytes"


@pytest.fixture
def service(monkeypatch) -> _StubService:
    stub = _StubService()
    monkeypatch.setattr(public_media, "_service", stub)
    return stub


@pytest.fixture
def charges(monkeypatch) -> list[list[str]]:
    recorded: list[list[str]] = []

    async def record(keys, *, limit, window_seconds, detail):
        recorded.append(list(keys))

    # The throttle moved out of `public_media` into the shared
    # `public_rate_limit` module, so that is where the name now resolves.
    monkeypatch.setattr(public_rate_limit, "consume_rate_limit", record)
    return recorded


@pytest.fixture
def client(service) -> TestClient:
    app = FastAPI()
    app.include_router(public_media.router)
    app.dependency_overrides[get_db] = lambda: None
    return TestClient(app)


def _url(width: int | str | None = None) -> str:
    base = f"/public/media/parts/{PART_ID}"
    return base if width is None else f"{base}?w={width}"


@pytest.mark.parametrize("width", [0, 1, 199, 201, 640, 2048, 2049, -1])
def test_widths_outside_the_allowlist_are_rejected(client, charges, width) -> None:
    assert width not in PUBLIC_THUMBNAIL_WIDTHS
    assert client.get(_url(width)).status_code == 422
    assert charges == []


@pytest.mark.parametrize("width", PUBLIC_THUMBNAIL_WIDTHS)
def test_allowlisted_widths_are_served_and_charged(client, service, charges, width) -> None:
    response = client.get(_url(width))

    assert response.status_code == 200
    assert service.reads == [width]
    assert charges == [["public-thumbnail:testclient"]]


def test_full_size_reads_are_not_charged_against_the_thumbnail_throttle(
    client, service, charges
) -> None:
    assert client.get(_url()).status_code == 200
    assert service.reads == [None]
    assert charges == []


def test_revalidated_thumbnails_are_not_charged(client, service, charges) -> None:
    etag = client.get(_url(200)).headers["etag"]
    charges.clear()
    service.reads.clear()

    conditional = client.get(_url(200), headers={"If-None-Match": etag})

    assert conditional.status_code == 304
    # A 304 renders nothing, so spending throttle budget on it would punish the readers
    # whose browsers cache correctly.
    assert charges == []
    assert service.reads == []
