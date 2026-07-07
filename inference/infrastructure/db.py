"""Sync SQLAlchemy engine for the ML-owned job queue."""

from __future__ import annotations

import re
import select

import psycopg2
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from inference.infrastructure.settings import get_inference_settings

convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=convention)


_settings = get_inference_settings()
engine = create_engine(_settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


_POSTGRES_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_postgres_identifier(identifier: str) -> str:
    if not _POSTGRES_IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"invalid PostgreSQL identifier: {identifier!r}")
    return f'"{identifier}"'


class JobNotificationListener:
    """Blocking LISTEN/NOTIFY helper for waking idle inference workers."""

    def __init__(self, channel: str | None = None) -> None:
        self.channel = channel or get_inference_settings().worker_notify_channel
        self._connection = None
        self._cursor = None

    def __enter__(self) -> JobNotificationListener:
        self._connection = psycopg2.connect(get_inference_settings().database_url)
        self._connection.autocommit = True
        self._cursor = self._connection.cursor()
        self._cursor.execute(f"LISTEN {_quote_postgres_identifier(self.channel)}")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def wait(self, timeout_seconds: float | None) -> list[str]:
        """Wait for job notifications and return any payloads received."""
        if self._connection is None:
            raise RuntimeError("JobNotificationListener must be opened before wait()")

        if timeout_seconds is None:
            readable, _, _ = select.select([self._connection], [], [])
        else:
            readable, _, _ = select.select([self._connection], [], [], timeout_seconds)
        if not readable:
            return []

        self._connection.poll()
        payloads: list[str] = []

        while self._connection.notifies:
            payloads.append(self._connection.notifies.pop(0).payload)

        return payloads

    def close(self) -> None:
        if self._cursor is not None:
            self._cursor.execute(f"UNLISTEN {_quote_postgres_identifier(self.channel)}")
            self._cursor.close()
            self._cursor = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
