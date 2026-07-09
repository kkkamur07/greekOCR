"""Platform job status notifications: Postgres NOTIFY detection + local SSE fan-out."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any

import asyncpg
from sqlalchemy import text

from backend.core.settings import get_infrastructure_settings
from backend.core.settings.job import get_job_settings
from backend.jobs.infrastructure.orm_models import JobStatus
from infrastructure.db import SyncSessionLocal

logger = logging.getLogger(__name__)

_POSTGRES_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_postgres_identifier(identifier: str) -> str:
    if not _POSTGRES_IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"invalid PostgreSQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _status_value(status: JobStatus | str) -> str:
    return status.value if isinstance(status, JobStatus) else status


def notify_platform_job_status_changed(
    job_id: uuid.UUID,
    status: JobStatus | str,
) -> None:
    """Emit a committed platform job status change to Postgres listeners."""
    payload = json.dumps({"job_id": str(job_id), "status": _status_value(status)})
    try:
        with SyncSessionLocal() as session:
            session.execute(
                text("SELECT pg_notify(:channel, :payload)"),
                {
                    "channel": get_job_settings().platform_job_notify_channel,
                    "payload": payload,
                },
            )
            session.commit()
    except Exception:
        logger.exception("failed to notify platform job status change for job %s", job_id)


class JobStatusBroadcaster:
    """Per-process registry of SSE queues waiting for job status changes."""

    def __init__(self) -> None:
        self._subscribers: dict[uuid.UUID, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: uuid.UUID) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=16)
        async with self._lock:
            self._subscribers.setdefault(job_id, set()).add(queue)
        return queue

    async def unsubscribe(
        self,
        job_id: uuid.UUID,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        async with self._lock:
            queues = self._subscribers.get(job_id)
            if queues is None:
                return
            queues.discard(queue)
            if not queues:
                self._subscribers.pop(job_id, None)

    async def publish(self, job_id: uuid.UUID, payload: dict[str, Any]) -> None:
        async with self._lock:
            queues = tuple(self._subscribers.get(job_id, ()))
        for queue in queues:
            if queue.full():
                _ = queue.get_nowait()
            queue.put_nowait(payload)


job_status_broadcaster = JobStatusBroadcaster()


def _parse_notification_payload(payload: str) -> tuple[uuid.UUID, dict[str, Any]] | None:
    try:
        data = json.loads(payload)
        job_id = uuid.UUID(str(data["job_id"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Ignoring invalid platform job notification payload: %r", payload)
        return None
    return job_id, data


async def platform_job_notification_loop(
    stop_event: asyncio.Event,
    broadcaster: JobStatusBroadcaster = job_status_broadcaster,
) -> None:
    """Forward Postgres NOTIFY payloads into this process' SSE subscribers."""
    channel = get_job_settings().platform_job_notify_channel
    _quote_postgres_identifier(channel)

    while not stop_event.is_set():
        connection: asyncpg.Connection | None = None

        async def publish_notification(raw_payload: str) -> None:
            parsed = _parse_notification_payload(raw_payload)
            if parsed is None:
                return
            job_id, payload = parsed
            await broadcaster.publish(job_id, payload)

        def handle_notification(
            _connection: asyncpg.Connection,
            _pid: int,
            _channel: str,
            raw_payload: str,
        ) -> None:
            asyncio.create_task(publish_notification(raw_payload))

        try:
            connection = await asyncpg.connect(
                get_infrastructure_settings().sync_database_url
            )
            await connection.add_listener(channel, handle_notification)
            logger.info("Listening for platform job notifications on %s", channel)
            await stop_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("platform job notification listener failed")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=1.0)
            except TimeoutError:
                pass
        finally:
            if connection is not None:
                try:
                    await connection.remove_listener(channel, handle_notification)
                finally:
                    await connection.close()
