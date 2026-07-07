"""Add auth_rate_limit_attempts table for shared-state brute-force protection

Revision ID: 011_auth_rate_limit
Revises: 010_jobs_payload_gin
Create Date: 2026-06-18

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "011_auth_rate_limit"
down_revision: Union[str, None] = "010_jobs_payload_gin"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "auth_rate_limit_attempts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(255), nullable=False),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_auth_rate_limit_attempts"),
    )
    op.create_index(
        "ix_auth_rate_limit_key_time",
        "auth_rate_limit_attempts",
        ["key", "attempted_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_auth_rate_limit_key_time", table_name="auth_rate_limit_attempts")
    op.drop_table("auth_rate_limit_attempts")
