"""add missing fk indexes and list pagination composites

Revision ID: 022_fk_and_pagination_indexes
Revises: 021_drop_rls
Create Date: 2026-07-09

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "022_fk_and_pagination_indexes"
down_revision: str | None = "021_drop_rls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(op.f("ix_jobs_binding_id"), "jobs", ["binding_id"], unique=False)
    op.create_index(op.f("ix_jobs_document_part_id"), "jobs", ["document_part_id"], unique=False)
    op.create_index(op.f("ix_jobs_model_id"), "jobs", ["model_id"], unique=False)
    op.create_index(
        op.f("ix_transcriptions_created_by_job_id"),
        "transcriptions",
        ["created_by_job_id"],
        unique=False,
    )
    op.create_index(
        "ix_documents_project_created_id",
        "documents",
        ["project_id", sa.text("created_at DESC"), sa.text("id DESC")],
        unique=False,
        postgresql_where=sa.text("workflow != 'archived'"),
    )
    op.create_index(
        "ix_projects_owner_created_id",
        "projects",
        ["owner_id", sa.text("created_at DESC"), sa.text("id DESC")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_projects_owner_created_id", table_name="projects")
    op.drop_index("ix_documents_project_created_id", table_name="documents")
    op.drop_index(op.f("ix_transcriptions_created_by_job_id"), table_name="transcriptions")
    op.drop_index(op.f("ix_jobs_model_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_document_part_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_binding_id"), table_name="jobs")
