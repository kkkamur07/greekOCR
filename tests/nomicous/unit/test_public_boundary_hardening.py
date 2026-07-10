"""Unit coverage for public-boundary input limits."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from backend.core.api.pagination import (
    MAX_CURSOR_LENGTH,
    PageCursor,
    PageResponse,
    decode_cursor,
    encode_cursor,
)
from backend.core.exceptions import ValidationError
from backend.users.application.password import (
    MAX_BCRYPT_PASSWORD_BYTES,
    hash_password,
    verify_password,
)


@pytest.mark.parametrize(
    "cursor",
    [
        "",
        "!",
        "%%%=",
        "eyJjcmVhdGVkX2F0Ijoibm90LWEtdGltZSIsImlkIjoibm90LWEtdXVpZCJ9",
        "a" * (MAX_CURSOR_LENGTH + 1),
    ],
)
def test_decode_cursor_rejects_malformed_or_unbounded_input(cursor: str) -> None:
    with pytest.raises(ValidationError, match="Invalid pagination cursor"):
        decode_cursor(cursor)


def test_decode_cursor_accepts_its_own_bounded_encoding() -> None:
    row_id = uuid.uuid4()
    created_at = datetime.now(UTC)

    decoded = decode_cursor(encode_cursor(created_at, row_id))

    assert decoded.id == row_id
    assert decoded.created_at == created_at


def test_cursor_pagination_types_remain_importable() -> None:
    cursor = PageCursor(created_at=datetime.now(UTC), id=uuid.uuid4())
    page = PageResponse[PageCursor](items=[cursor])

    assert page.items == [cursor]
    assert page.next_cursor is None


def test_bcrypt_password_limit_is_measured_in_utf8_bytes() -> None:
    accepted = "é" * 36
    rejected = "é" * 37
    assert len(accepted.encode("utf-8")) == MAX_BCRYPT_PASSWORD_BYTES

    hashed = hash_password(accepted)
    assert verify_password(accepted, hashed)

    with pytest.raises(ValueError, match="bcrypt UTF-8 byte limit"):
        hash_password(rejected)
    assert not verify_password(rejected, hashed)
