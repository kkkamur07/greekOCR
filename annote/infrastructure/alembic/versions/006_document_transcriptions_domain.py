"""document domain (transcriptions) — Transcription, LineTranscription

Revision ID: 006_document_transcriptions
Revises: 005_inference_jobs
Create Date: 2026-05-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "006_document_transcriptions"
down_revision: Union[str, None] = "005_inference_jobs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

transcription_kind = postgresql.ENUM(
    "ground_truth", "model", name="transcription_kind", create_type=False
)


def upgrade() -> None:
    transcription_kind.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "transcriptions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("kind", transcription_kind, nullable=False),
        sa.Column("created_by_job_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["created_by_job_id"],
            ["jobs.id"],
            name=op.f("fk_transcriptions_created_by_job_id_jobs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_transcriptions_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transcriptions")),
    )
    op.create_index(op.f("ix_transcriptions_document_id"), "transcriptions", ["document_id"], unique=False)
    op.create_index(
        "uq_transcriptions_one_ground_truth",
        "transcriptions",
        ["document_id"],
        unique=True,
        postgresql_where="kind = 'ground_truth'",
    )
    op.create_table(
        "line_transcriptions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("line_id", sa.UUID(), nullable=False),
        sa.Column("transcription_id", sa.UUID(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["line_id"],
            ["lines.id"],
            name=op.f("fk_line_transcriptions_line_id_lines"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["transcription_id"],
            ["transcriptions.id"],
            name=op.f("fk_line_transcriptions_transcription_id_transcriptions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_line_transcriptions")),
        sa.UniqueConstraint("line_id", "transcription_id", name="uq_line_transcriptions_line_layer"),
    )
    op.create_index(op.f("ix_line_transcriptions_line_id"), "line_transcriptions", ["line_id"], unique=False)
    op.create_index(
        op.f("ix_line_transcriptions_transcription_id"),
        "line_transcriptions",
        ["transcription_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_line_transcriptions_transcription_id"), table_name="line_transcriptions")
    op.drop_index(op.f("ix_line_transcriptions_line_id"), table_name="line_transcriptions")
    op.drop_table("line_transcriptions")
    op.drop_index(
        "uq_transcriptions_one_ground_truth",
        table_name="transcriptions",
        postgresql_where="kind = 'ground_truth'",
    )
    op.drop_index(op.f("ix_transcriptions_document_id"), table_name="transcriptions")
    op.drop_table("transcriptions")
    transcription_kind.drop(op.get_bind(), checkfirst=True)
