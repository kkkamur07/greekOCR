"""Create auth_rate_limit_attempts for shared auth throttling."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "004_auth_rate_limit_attempts"
down_revision: str | None = "003_job_status_cancelled"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "auth_rate_limit_attempts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_auth_rate_limit_key_time",
        "auth_rate_limit_attempts",
        ["key", "attempted_at"],
        unique=False,
    )
    # 002 granted ALL TABLES at migration time; new tables need explicit grants.
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomicous_api') THEN
            GRANT SELECT, INSERT, UPDATE, DELETE ON auth_rate_limit_attempts TO nomicous_api;
            GRANT USAGE, SELECT ON SEQUENCE auth_rate_limit_attempts_id_seq TO nomicous_api;
          END IF;
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomicous_migrator') THEN
            GRANT ALL PRIVILEGES ON auth_rate_limit_attempts TO nomicous_migrator;
            GRANT ALL PRIVILEGES ON SEQUENCE auth_rate_limit_attempts_id_seq TO nomicous_migrator;
          END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.drop_index("ix_auth_rate_limit_key_time", table_name="auth_rate_limit_attempts")
    op.drop_table("auth_rate_limit_attempts")
