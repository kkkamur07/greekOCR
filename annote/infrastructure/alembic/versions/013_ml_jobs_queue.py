"""add ML-owned jobs queue

Revision ID: 013_ml_jobs_queue
Revises: 012_ml_job_callbacks
Create Date: 2026-07-04

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "013_ml_jobs_queue"
down_revision: str | None = "012_ml_job_callbacks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ml_task = postgresql.ENUM("segment", "transcribe", name="ml_task", create_type=False)
ml_job_status = postgresql.ENUM(
    "pending",
    "running",
    "done",
    "failed",
    name="ml_job_status",
    create_type=False,
)


def _create_ml_jobs_table() -> None:
    op.create_table(
        "ml_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("product_job_id", sa.UUID(), nullable=False),
        sa.Column("task", ml_task, nullable=False),
        sa.Column("registry_model_id", sa.Text(), nullable=False),
        sa.Column("registry_tag", sa.Text(), nullable=False),
        sa.Column("status", ml_job_status, nullable=False),
        sa.Column("image_bytes", sa.LargeBinary(), nullable=False),
        sa.Column(
            "params",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ml_jobs")),
    )


def _create_missing_ml_jobs_indexes(bind) -> None:
    ml_jobs = sa.Table(
        "ml_jobs",
        sa.MetaData(),
        sa.Column("product_job_id", sa.UUID()),
        sa.Column("status", ml_job_status),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("id", sa.UUID()),
    )
    sa.Index(op.f("ix_ml_jobs_product_job_id"), ml_jobs.c.product_job_id).create(
        bind,
        checkfirst=True,
    )
    sa.Index(op.f("ix_ml_jobs_status"), ml_jobs.c.status).create(bind, checkfirst=True)
    sa.Index(
        "ix_ml_jobs_claim_order",
        ml_jobs.c.status,
        ml_jobs.c.created_at,
        ml_jobs.c.id,
    ).create(bind, checkfirst=True)


def _ml_jobs_table_exists(bind) -> bool:
    return sa.inspect(bind).has_table("ml_jobs")


def upgrade() -> None:
    bind = op.get_bind()
    ml_task.create(bind, checkfirst=True)
    ml_job_status.create(bind, checkfirst=True)

    if not _ml_jobs_table_exists(bind):
        _create_ml_jobs_table()
    _create_missing_ml_jobs_indexes(bind)


def downgrade() -> None:
    bind = op.get_bind()
    op.drop_index("ix_ml_jobs_claim_order", table_name="ml_jobs")
    op.drop_index(op.f("ix_ml_jobs_status"), table_name="ml_jobs")
    op.drop_index(op.f("ix_ml_jobs_product_job_id"), table_name="ml_jobs")
    op.drop_table("ml_jobs")
    ml_job_status.drop(bind, checkfirst=True)
    ml_task.drop(bind, checkfirst=True)
