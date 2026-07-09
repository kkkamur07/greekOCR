"""Keyset cursor pagination helpers."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

T = TypeVar("T")


class PageParams(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)
    cursor: str | None = None


class PageCursor(BaseModel):
    created_at: datetime
    id: UUID


class PageResponse(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None = None


def encode_cursor(created_at: datetime, row_id: UUID) -> str:
    payload = {
        "created_at": created_at.isoformat(),
        "id": str(row_id),
    }
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def decode_cursor(cursor: str) -> PageCursor:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
        return PageCursor(
            created_at=datetime.fromisoformat(payload["created_at"]),
            id=UUID(payload["id"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid pagination cursor") from exc


def paginate_rows(
    rows: list[T],
    *,
    limit: int,
    created_at_getter,
    id_getter,
) -> tuple[list[T], str | None]:
    """Slice *limit + 1* rows and emit next cursor when more pages exist."""
    has_more = len(rows) > limit
    page = rows[:limit]
    if not has_more or not page:
        return page, None
    last = page[-1]
    return page, encode_cursor(created_at_getter(last), id_getter(last))
