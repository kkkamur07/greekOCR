"""add query performance indexes

Revision ID: 014_query_performance_indexes
Revises: 013_ml_jobs_queue
Create Date: 2026-07-09

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "014_query_performance_indexes"
down_revision: str | None = "013_ml_jobs_queue"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_jobs_claim_pending",
        "jobs",
        ["created_at", "id"],
        unique=False,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(op.f("ix_jobs_document_id"), "jobs", ["document_id"], unique=False)
    op.create_index(op.f("ix_projects_owner_id"), "projects", ["owner_id"], unique=False)
    op.create_index(
        op.f("ix_project_shared_users_user_id"),
        "project_shared_users",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_document_parts_document_order",
        "document_parts",
        ["document_id", "order"],
        unique=False,
    )
    op.create_index(
        "ix_lines_part_order",
        "lines",
        ["part_id", "order", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_lines_part_order", table_name="lines")
    op.drop_index("ix_document_parts_document_order", table_name="document_parts")
    op.drop_index(op.f("ix_project_shared_users_user_id"), table_name="project_shared_users")
    op.drop_index(op.f("ix_projects_owner_id"), table_name="projects")
    op.drop_index(op.f("ix_jobs_document_id"), table_name="jobs")
    op.drop_index("ix_jobs_claim_pending", table_name="jobs")
