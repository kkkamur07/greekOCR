"""Add ML job callback fields to jobs

Revision ID: 012_ml_job_callbacks
Revises: 011_auth_rate_limit
Create Date: 2026-07-04

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "012_ml_job_callbacks"
down_revision: str | None = "011_auth_rate_limit"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TYPE job_status ADD VALUE IF NOT EXISTS 'waiting'")
    op.add_column("jobs", sa.Column("inference_job_id", sa.UUID(), nullable=True))
    op.create_index(op.f("ix_jobs_inference_job_id"), "jobs", ["inference_job_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_jobs_inference_job_id"), table_name="jobs")
    op.drop_column("jobs", "inference_job_id")
    # PostgreSQL does not support dropping enum values without recreating the type.
