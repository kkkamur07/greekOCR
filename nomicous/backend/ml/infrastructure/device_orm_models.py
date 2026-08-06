"""Paired helper devices and the one-time pairing requests that create them.

A *helper device* is a researcher's own computer. It is authorised once from a
logged-in browser and thereafter authenticates outbound with an opaque device
token, so the browser never has to reach loopback.

Only credential *hashes* are stored here, exactly as ``auth_sessions`` does for
browser sessions - the raw token exists for the length of one HTTP response and
is never written to Postgres, a log line, or a read endpoint.

Two tables rather than one nullable table: folding pairings into
``helper_devices`` would force ``user_id`` to be nullable and destroy the
database-level invariant that a device belongs to exactly one researcher. Every
device query would then have to remember ``AND approved_at IS NOT NULL``, and
one forgotten clause is an authentication bypass.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Index, SmallInteger, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.ml.domain.execution import ExecutionTarget
from backend.users.infrastructure.orm_models import User
from infrastructure.db import Base

MAX_DEVICE_NAME_LENGTH = 120
MAX_PLATFORM_LENGTH = 32
MAX_HELPER_VERSION_LENGTH = 32
MAX_IP_LENGTH = 45
MAX_USER_AGENT_LENGTH = 255
HASH_LENGTH = 64


class HelperDevice(Base):
    """One researcher-owned computer authorised to run that researcher's jobs."""

    __tablename__ = "helper_devices"
    __table_args__ = (
        # Partial index: the only device lookup that is not by primary key is
        # "the live devices of this user".
        Index("ix_helper_devices_user_live", "user_id", postgresql_where="revoked_at IS NULL"),
        Index("ix_helper_devices_last_seen_at", "last_seen_at"),
    )

    # Also the token's lookup key: the wire token carries this id, so
    # authentication is a primary-key fetch plus one constant-time compare and
    # never a scan over ``token_hash``. That is why there is no index on the
    # digest column - we never search by it.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # The entire authorization scope of the credential. NOT NULL by design.
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(MAX_DEVICE_NAME_LENGTH), nullable=False)
    # Which **inference host** this device *is*. A hosted worker is a device like
    # any other (ADR 0003), so **capacity** is one query over one table rather
    # than a device check plus a separate notion of "is cloud up". Everything
    # paired through the browser is a researcher's own computer; ``cloud`` rows
    # are provisioned for hosted workers.
    inference_host: Mapped[ExecutionTarget] = mapped_column(
        Enum(ExecutionTarget, name="execution_target"),
        nullable=False,
        server_default=ExecutionTarget.local.value,
        default=ExecutionTarget.local,
    )
    platform: Mapped[str] = mapped_column(String(MAX_PLATFORM_LENGTH), nullable=False)
    helper_version: Mapped[str] = mapped_column(String(MAX_HELPER_VERSION_LENGTH), nullable=False)
    capabilities: Mapped[dict] = mapped_column(
        JSONB, default=dict, server_default="{}", nullable=False
    )
    # ``""`` means approved-but-not-collected. An empty string can never equal a
    # 64-character hex digest, so a half-finished pairing cannot authenticate.
    token_hash: Mapped[str] = mapped_column(String(HASH_LENGTH), nullable=False, server_default="")
    previous_token_hash: Mapped[str | None] = mapped_column(String(HASH_LENGTH), nullable=True)
    previous_token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Log-safe correlation handle, e.g. ``nmd1.4f3a9c81``. Never the secret.
    token_prefix: Mapped[str] = mapped_column(String(20), nullable=False, server_default="")
    token_issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paired_from_ip: Mapped[str | None] = mapped_column(String(MAX_IP_LENGTH), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_ip: Mapped[str | None] = mapped_column(String(MAX_IP_LENGTH), nullable=True)
    # Soft delete: jobs that reference this device keep resolving after revocation.
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped[User] = relationship("User")


class HelperPairing(Base):
    """A single-use, short-lived authorisation request from an unpaired helper."""

    __tablename__ = "helper_pairings"
    # One index, on expires_at: it serves the live-pairing count and the sweep of
    # finished rows. There is deliberately no index on request_ip - nothing
    # queries by it, because behind an unallowlisted proxy that column holds the
    # same address for every row.
    __table_args__ = (Index("ix_helper_pairings_expires_at", "expires_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Held by the helper; proves "I am the process that started this pairing".
    device_code_hash: Mapped[str] = mapped_column(String(HASH_LENGTH), nullable=False)
    # Held by the browser; arrives in the URL *fragment* and is submitted in a
    # POST body, so it never reaches an access log, a Referer, or history.
    verification_token_hash: Mapped[str] = mapped_column(
        String(HASH_LENGTH), nullable=False, unique=True
    )
    requested_name: Mapped[str] = mapped_column(String(MAX_DEVICE_NAME_LENGTH), nullable=False)
    requested_platform: Mapped[str] = mapped_column(String(MAX_PLATFORM_LENGTH), nullable=False)
    requested_helper_version: Mapped[str] = mapped_column(
        String(MAX_HELPER_VERSION_LENGTH), nullable=False
    )
    requested_capabilities: Mapped[dict] = mapped_column(
        JSONB, default=dict, server_default="{}", nullable=False
    )
    request_ip: Mapped[str] = mapped_column(String(MAX_IP_LENGTH), nullable=False)
    request_user_agent: Mapped[str | None] = mapped_column(
        String(MAX_USER_AGENT_LENGTH), nullable=True
    )
    attempts: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    poll_interval_seconds: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="5"
    )
    delivery_count: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    approved_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    denied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    device_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("helper_devices.id", ondelete="CASCADE"), nullable=True
    )
