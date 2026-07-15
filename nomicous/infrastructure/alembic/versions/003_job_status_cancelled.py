"""Add cancelled to product job_status enum."""

from collections.abc import Sequence

from alembic import op

revision: str = "003_job_status_cancelled"
down_revision: str | None = "002_service_roles"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TYPE job_status ADD VALUE IF NOT EXISTS 'cancelled'")


def downgrade() -> None:
    # PostgreSQL cannot drop enum values safely; leave cancelled in place.
    pass
