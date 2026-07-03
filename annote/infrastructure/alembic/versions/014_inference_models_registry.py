"""Replace inference_models.artifact_ref with registry id + tag

Revision ID: 014_inference_models_registry
Revises: 011_auth_rate_limit
Create Date: 2026-07-03

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "014_inference_models_registry"
down_revision: Union[str, None] = "011_auth_rate_limit"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "inference_models",
        sa.Column("registry_model_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "inference_models",
        sa.Column(
            "registry_tag",
            sa.String(length=64),
            server_default="stable",
            nullable=False,
        ),
    )
    op.execute(
        """
        UPDATE inference_models
        SET registry_model_id = CASE task::text
            WHEN 'segment' THEN 'kraken-blla'
            WHEN 'transcribe' THEN 'greek-calamariv1'
            ELSE name
        END
        """
    )
    op.alter_column("inference_models", "registry_model_id", nullable=False)
    op.drop_column("inference_models", "artifact_ref")


def downgrade() -> None:
    op.add_column(
        "inference_models",
        sa.Column("artifact_ref", sa.String(length=1024), nullable=True),
    )
    op.execute(
        """
        UPDATE inference_models
        SET artifact_ref = registry_model_id
        """
    )
    op.alter_column("inference_models", "artifact_ref", nullable=False)
    op.drop_column("inference_models", "registry_tag")
    op.drop_column("inference_models", "registry_model_id")
