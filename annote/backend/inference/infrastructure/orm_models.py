from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from infrastructure.db import Base

if TYPE_CHECKING:
    from backend.document.infrastructure.orm_models import Document, DocumentPart
    from backend.project.infrastructure.orm_models import Project


class InferenceTask(str, enum.Enum):
    segment = "segment"
    transcribe = "transcribe"
    binarize = "binarize"


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class JobType(str, enum.Enum):
    segment = "segment"
    transcribe = "transcribe"
    binarize = "binarize"
    pipeline = "pipeline"


class InferenceModel(Base):
    __tablename__ = "inference_models"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    provider: Mapped[str] = mapped_column(String(64))
    task: Mapped[InferenceTask] = mapped_column(Enum(InferenceTask, name="inference_task"))
    artifact_ref: Mapped[str] = mapped_column(String(1024))
    default_params: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    bindings: Mapped[list[ModelBinding]] = relationship("ModelBinding", back_populates="model")
    jobs: Mapped[list[Job]] = relationship("Job", back_populates="model")


class ModelBinding(Base):
    __tablename__ = "model_bindings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task: Mapped[InferenceTask] = mapped_column(Enum(InferenceTask, name="binding_task"))
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("inference_models.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True
    )
    document_part_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_parts.id", ondelete="CASCADE"), nullable=True, index=True
    )
    overrides: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    model: Mapped[InferenceModel] = relationship("InferenceModel", back_populates="bindings")
    project: Mapped[Project | None] = relationship("Project", back_populates="model_bindings")
    document: Mapped[Document | None] = relationship("Document", back_populates="model_bindings")
    document_part: Mapped[DocumentPart | None] = relationship(
        "DocumentPart", back_populates="model_bindings"
    )
    jobs: Mapped[list[Job]] = relationship("Job", back_populates="binding")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type: Mapped[JobType] = mapped_column(Enum(JobType, name="job_type"))
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"), default=JobStatus.pending, index=True
    )
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("inference_models.id", ondelete="SET NULL"), nullable=True
    )
    binding_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_bindings.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    document_part_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_parts.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model: Mapped[InferenceModel | None] = relationship("InferenceModel", back_populates="jobs")
    binding: Mapped[ModelBinding | None] = relationship("ModelBinding", back_populates="jobs")
