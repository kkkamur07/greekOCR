"""ML-owned Postgres models — only the ML service reads/writes these tables."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, LargeBinary, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ml.contracts.common import MLJobStatus, MLTask
from ml.infrastructure.db import Base


class MLJob(Base):
    __tablename__ = "ml_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    task: Mapped[MLTask] = mapped_column(Enum(MLTask, name="ml_task"))
    registry_model_id: Mapped[str] = mapped_column(Text)
    registry_tag: Mapped[str] = mapped_column(Text, default="stable")
    status: Mapped[MLJobStatus] = mapped_column(
        Enum(MLJobStatus, name="ml_job_status"),
        default=MLJobStatus.pending,
        index=True,
    )
    image_bytes: Mapped[bytes] = mapped_column(LargeBinary)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
