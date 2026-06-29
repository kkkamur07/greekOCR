"""add GIN index for jobs payload containment queries

Revision ID: 010_jobs_payload_gin
Revises: 009_annotation_history
Create Date: 2026-06-18

"""

from typing import Sequence, Union

from alembic import op

revision: str = "010_jobs_payload_gin"
down_revision: Union[str, None] = "009_annotation_history"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_jobs_payload_gin",
        "jobs",
        ["payload"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_jobs_payload_gin", table_name="jobs")
