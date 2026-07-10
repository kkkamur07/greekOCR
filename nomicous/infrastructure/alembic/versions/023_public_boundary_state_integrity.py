"""Harden public errors and durable backend state transitions.

Revision ID: 023_boundary_state
Revises: 022_fk_and_pagination_indexes
Create Date: 2026-07-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "023_boundary_state"
down_revision: str | None = "022_fk_and_pagination_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("callback_claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        """
        WITH ordered AS (
          SELECT id,
                 row_number() OVER (
                   PARTITION BY document_id
                   ORDER BY "order", created_at, id
                 ) - 1 AS new_order
          FROM document_parts
        )
        UPDATE document_parts AS parts
        SET "order" = ordered.new_order
        FROM ordered
        WHERE parts.id = ordered.id
        """
    )
    op.create_unique_constraint(
        "uq_document_parts_document_order",
        "document_parts",
        ["document_id", "order"],
    )
    op.create_table(
        "media_deletion_intents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("image_key", sa.String(length=1024), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_media_deletion_intents"),
        sa.UniqueConstraint("image_key", name="uq_media_deletion_intents_image_key"),
    )
    op.create_index(
        "ix_media_deletion_intents_pending",
        "media_deletion_intents",
        ["created_at"],
        postgresql_where=sa.text("completed_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_media_deletion_intents_pending", table_name="media_deletion_intents")
    op.drop_table("media_deletion_intents")
    op.drop_constraint("uq_document_parts_document_order", "document_parts", type_="unique")
    op.drop_column("jobs", "callback_claimed_at")
