"""line transcription text_source and character_confidences

Revision ID: 012_line_tx_source
Revises: 011_auth_rate_limit
Create Date: 2026-07-03

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "012_line_tx_source"
down_revision: Union[str, None] = "011_auth_rate_limit"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

line_transcription_text_source = postgresql.ENUM(
    "model",
    "human_edited",
    name="line_transcription_text_source",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    line_transcription_text_source.create(bind, checkfirst=True)

    op.add_column(
        "line_transcriptions",
        sa.Column(
            "text_source",
            line_transcription_text_source,
            server_default="human_edited",
            nullable=False,
        ),
    )
    op.add_column(
        "line_transcriptions",
        sa.Column(
            "character_confidences",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )

    # Model-layer rows from OCR jobs keep model provenance; GT stays human_edited.
    op.execute(
        """
        UPDATE line_transcriptions AS lt
        SET text_source = 'model'
        FROM transcriptions AS t
        WHERE lt.transcription_id = t.id
          AND t.kind = 'model'
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    op.drop_column("line_transcriptions", "character_confidences")
    op.drop_column("line_transcriptions", "text_source")
    line_transcription_text_source.drop(bind, checkfirst=True)
