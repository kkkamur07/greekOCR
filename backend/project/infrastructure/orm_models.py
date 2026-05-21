import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Table, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.infrastructure.db import Base

project_shared_users = Table(
    "project_shared_users",
    Base.metadata,
    Column("project_id", UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(512))
    slug: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    guidelines: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    owner: Mapped["User | None"] = relationship(back_populates="owned_projects")
    shared_users: Mapped[list["User"]] = relationship(
        secondary=project_shared_users,
        back_populates="shared_projects",
    )
    documents: Mapped[list["Document"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    model_bindings: Mapped[list["ModelBinding"]] = relationship(back_populates="project")
