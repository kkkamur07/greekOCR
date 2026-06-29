"""annotation history snapshots

Revision ID: 009_annotation_history
Revises: 008_page_tx_lines
Create Date: 2026-06-16

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "009_annotation_history"
down_revision: Union[str, None] = "008_page_tx_lines"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "annotation_history_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("part_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("state", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("line_count", sa.Integer(), nullable=False),
        sa.Column("paired_line_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["part_id"], ["document_parts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_annotation_history_snapshots_part_id",
        "annotation_history_snapshots",
        ["part_id"],
    )
    op.create_index(
        "ix_annotation_history_snapshots_part_created",
        "annotation_history_snapshots",
        ["part_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_annotation_history_snapshots_part_created", table_name="annotation_history_snapshots")
    op.drop_index("ix_annotation_history_snapshots_part_id", table_name="annotation_history_snapshots")
    op.drop_table("annotation_history_snapshots")
