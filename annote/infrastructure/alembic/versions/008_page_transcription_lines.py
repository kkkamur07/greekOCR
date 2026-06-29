"""page transcription helper lines

Revision ID: 008_page_tx_lines
Revises: 007_doc_line_tx
Create Date: 2026-06-16

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "008_page_tx_lines"
down_revision: Union[str, None] = "007_doc_line_tx"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "page_transcription_lines",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("part_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("paired_line_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["paired_line_id"], ["lines.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["part_id"], ["document_parts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("paired_line_id", name="uq_page_transcription_lines_paired_line"),
        sa.UniqueConstraint("part_id", "order", name="uq_page_transcription_lines_part_order"),
    )
    op.create_index("ix_page_transcription_lines_part_id", "page_transcription_lines", ["part_id"])
    op.create_index(
        "ix_page_transcription_lines_paired_line_id",
        "page_transcription_lines",
        ["paired_line_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_page_transcription_lines_paired_line_id", table_name="page_transcription_lines")
    op.drop_index("ix_page_transcription_lines_part_id", table_name="page_transcription_lines")
    op.drop_table("page_transcription_lines")
