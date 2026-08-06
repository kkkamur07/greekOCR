"""Add worker ownership columns to jobs.

``claimed_by`` records the worker process that owns a running job, so a job that
was reclaimed after its lease expired cannot be completed by the zombie worker
that lost it. ``heartbeat_at`` records that worker's last liveness signal.

This is a real change on every database, including a fresh one. It did not used
to be: ``001_initial_schema`` was regenerated from live ORM metadata, so a
database created from scratch already had both columns and this revision was a
no-op there. That is exactly what the squash froze 001 to stop - see its
docstring - and 001 now names ``jobs.claimed_by`` and ``jobs.heartbeat_at`` as
deliberately absent from the baseline.

The ``IF NOT EXISTS`` guards stay, and their only remaining subject is a database
stamped during the period 001 was still being regenerated: those already have the
columns and must not fail on a second ``ADD COLUMN``. On any database created
since the freeze, the guards never fire.
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
