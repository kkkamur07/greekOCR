"""Durable object-store deletion retries for document media."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import infrastructure.models  # noqa: F401 — register MediaDeletionIntent mapper
from sqlalchemy import select

from backend.document.infrastructure.media_store import get_media_store
from backend.document.infrastructure.orm_models import MediaDeletionIntent
from infrastructure.db import sync_system_session

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from asyncio import Event


def process_media_deletion_intents(*, batch_size: int = 50) -> int:
    """Delete a bounded batch of committed media intents."""
    completed = 0
    with sync_system_session() as session:
        intents = list(
            session.execute(
                select(MediaDeletionIntent)
                .where(MediaDeletionIntent.completed_at.is_(None))
                .order_by(MediaDeletionIntent.created_at, MediaDeletionIntent.id)
                .with_for_update(skip_locked=True)
                .limit(batch_size)
            ).scalars()
        )
        store = get_media_store()
        for intent in intents:
            try:
                store.delete(intent.image_key)
            except Exception as exc:
                intent.attempts += 1
                intent.last_error = f"{type(exc).__name__}: {str(exc)[:900]}"
                logger.warning(
                    "media deletion deferred intent_id=%s attempt=%s exception=%s",
                    intent.id,
                    intent.attempts,
                    type(exc).__name__,
                )
                continue
            intent.attempts += 1
            intent.last_error = None
            intent.completed_at = datetime.now(UTC)
            completed += 1
        session.commit()
    return completed


async def media_gc_loop(stop_event: Event, *, interval_seconds: float = 60.0) -> None:
    """Periodically retry durable media deletes without blocking request handlers."""
    import asyncio

    while not stop_event.is_set():
        try:
            await asyncio.to_thread(process_media_deletion_intents)
        except Exception:
            logger.exception("media deletion GC pass failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            pass
