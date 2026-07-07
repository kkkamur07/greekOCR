"""document domain (line transcription model) — review status and segment geometry

Revision ID: 007_doc_line_tx
Revises: 006_document_transcriptions
Create Date: 2026-06-16

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "007_doc_line_tx"
down_revision: Union[str, None] = "006_document_transcriptions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

line_geometry_kind = postgresql.ENUM(
    "polygon", "rectangle", name="line_geometry_kind", create_type=False
)
line_source = postgresql.ENUM("manual", "kraken", "model", name="line_source", create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    line_geometry_kind.create(bind, checkfirst=True)
    line_source.create(bind, checkfirst=True)

    op.add_column(
        "document_parts",
        sa.Column("reviewed", sa.Boolean(), server_default="false", nullable=False),
    )
    op.add_column(
        "lines",
        sa.Column(
            "kind",
            line_geometry_kind,
            server_default="polygon",
            nullable=False,
        ),
    )
    op.add_column(
        "lines",
        sa.Column(
            "points",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
    )
    op.add_column(
        "lines",
        sa.Column("source", line_source, server_default="manual", nullable=False),
    )
    op.add_column(
        "lines",
        sa.Column("source_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "lines",
        sa.Column("kraken_ceiling", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    bind = op.get_bind()
    op.drop_column("lines", "kraken_ceiling")
    op.drop_column("lines", "source_metadata")
    op.drop_column("lines", "source")
    op.drop_column("lines", "points")
    op.drop_column("lines", "kind")
    op.drop_column("document_parts", "reviewed")
    line_source.drop(bind, checkfirst=True)
    line_geometry_kind.drop(bind, checkfirst=True)
