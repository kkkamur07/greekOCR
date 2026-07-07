from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.project.infrastructure.orm_models import project_shared_users
from infrastructure.db import Base

if TYPE_CHECKING:
    from backend.project.infrastructure.orm_models import Project


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(150), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    owned_projects: Mapped[list[Project]] = relationship("Project", back_populates="owner")
    shared_projects: Mapped[list[Project]] = relationship(
        "Project",
        secondary=project_shared_users,
        back_populates="shared_users",
    )


class AuthRateLimitAttempt(Base):
    """One row per auth attempt; used by the shared-state rate limiter."""

    __tablename__ = "auth_rate_limit_attempts"
    __table_args__ = (
        Index("ix_auth_rate_limit_key_time", "key", "attempted_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
