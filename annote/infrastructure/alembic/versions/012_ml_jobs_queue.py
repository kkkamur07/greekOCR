"""add ML-owned jobs queue

Revision ID: 012_ml_jobs_queue
Revises: 011_auth_rate_limit
Create Date: 2026-07-04

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "012_ml_jobs_queue"
down_revision: Union[str, None] = "011_auth_rate_limit"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ml_task = postgresql.ENUM("segment", "transcribe", name="ml_task", create_type=False)
ml_job_status = postgresql.ENUM("pending", "running", "done", "failed", name="ml_job_status", create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    ml_task.create(bind, checkfirst=True)
    ml_job_status.create(bind, checkfirst=True)

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
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ml_jobs")),
    )
    op.create_index(op.f("ix_ml_jobs_product_job_id"), "ml_jobs", ["product_job_id"], unique=False)
    op.create_index(op.f("ix_ml_jobs_status"), "ml_jobs", ["status"], unique=False)
    op.create_index(
        "ix_ml_jobs_claim_order",
        "ml_jobs",
        ["status", "created_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    bind = op.get_bind()
    op.drop_index("ix_ml_jobs_claim_order", table_name="ml_jobs")
    op.drop_index(op.f("ix_ml_jobs_status"), table_name="ml_jobs")
    op.drop_index(op.f("ix_ml_jobs_product_job_id"), table_name="ml_jobs")
    op.drop_table("ml_jobs")
    ml_job_status.drop(bind, checkfirst=True)
    ml_task.drop(bind, checkfirst=True)
