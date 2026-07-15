"""Create the final application schema.

This is the squashed replacement for the pre-production migration history.
ORM metadata is the source of truth (includes cancelled job status,
jobs.callback_claimed_at, auth_rate_limit_attempts, etc.). Application
authorization remains in FastAPI; this schema intentionally does not enable
PostgreSQL row-level security.

Service role grants live in 002_service_roles.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from infrastructure.db import Base
from infrastructure import models  # noqa: F401 - register all ORM tables

revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_inference_jobs() -> None:
    bind = op.get_bind()
    inference_task = postgresql.ENUM(
        "segment",
        "transcribe",
        "binarize",
        name="inference_task",
        create_type=False,
    )
    inference_job_status = postgresql.ENUM(
        "pending",
        "running",
        "done",
        "failed",
        name="inference_job_status",
        create_type=False,
    )
    inference_job_status.create(bind, checkfirst=True)
    op.create_table(
        "inference_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("product_job_id", sa.UUID(), nullable=False),
        sa.Column("task", inference_task, nullable=False),
        sa.Column("registry_model_id", sa.Text(), nullable=False),
        sa.Column("registry_tag", sa.Text(), nullable=False),
        sa.Column("status", inference_job_status, nullable=False),
        sa.Column("image_bytes", sa.LargeBinary(), nullable=False),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_jobs")),
    )
    op.create_index(op.f("ix_inference_jobs_product_job_id"), "inference_jobs", ["product_job_id"])
    op.create_index(op.f("ix_inference_jobs_status"), "inference_jobs", ["status"])
    op.create_index(
        "ix_inference_jobs_claim_order",
        "inference_jobs",
        ["status", "created_at", "id"],
    )


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)
    _create_inference_jobs()


def downgrade() -> None:
    bind = op.get_bind()
    op.drop_index("ix_inference_jobs_claim_order", table_name="inference_jobs")
    op.drop_index(op.f("ix_inference_jobs_status"), table_name="inference_jobs")
    op.drop_index(op.f("ix_inference_jobs_product_job_id"), table_name="inference_jobs")
    op.drop_table("inference_jobs")
    Base.metadata.drop_all(bind=bind)
