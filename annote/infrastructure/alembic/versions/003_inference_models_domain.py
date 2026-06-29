"""inference domain (catalog) — InferenceModel

Revision ID: 003_inference_models
Revises: 002_project
Create Date: 2026-05-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003_inference_models"
down_revision: Union[str, None] = "002_project"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

inference_task = postgresql.ENUM(
    "segment", "transcribe", "binarize", name="inference_task", create_type=False
)


def upgrade() -> None:
    inference_task.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "inference_models",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("task", inference_task, nullable=False),
        sa.Column("artifact_ref", sa.String(length=1024), nullable=False),
        sa.Column(
            "default_params",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_models")),
        sa.UniqueConstraint("name", name=op.f("uq_inference_models_name")),
    )


def downgrade() -> None:
    op.drop_table("inference_models")
    inference_task.drop(op.get_bind(), checkfirst=True)
