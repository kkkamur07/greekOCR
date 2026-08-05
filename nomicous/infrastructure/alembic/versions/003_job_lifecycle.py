"""Add worker ownership columns to jobs.

``claimed_by`` records the worker process that owns a running job, so a job that
was reclaimed after its lease expired cannot be completed by the zombie worker
that lost it. ``heartbeat_at`` records that worker's last liveness signal.

``001_initial_schema`` builds the schema from live ORM metadata, so a database
created from scratch already has both columns. The statements below are written
idempotently so this migration is a no-op there and a real change on databases
stamped before the columns existed.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "003_job_lifecycle"
down_revision: str | None = "002_service_roles"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS claimed_by TEXT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS heartbeat_at")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS claimed_by")
