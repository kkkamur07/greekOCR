"""Count abandoned claims on a job, so a poison page cannot cycle forever.

``release_expired_device_leases`` and ``reclaim_stale_running_jobs`` both return
an abandoned job to ``pending``, deliberately: a closed laptop lid is not a
failed job, and the page should be picked up by the next agent. Nothing counted
how often that had already happened, so a page that reliably kills whatever runs
it - a corrupt scan, an image the model segfaults on - cycled
``pending -> waiting -> pending`` forever. It never reached a terminal status, so
the researcher was never told, and it consumed one agent slot on every lap.

This column is that counter. It is on ``jobs`` rather than in ``payload``
because the sweeps are bulk ``UPDATE`` statements over the queue and have to be
able to increment and compare it in SQL without reading a JSON document per row.

``NOT NULL DEFAULT 0`` backfills every existing row with the truthful value: no
claim on them has been counted, so none has been abandoned as far as the ceiling
is concerned. That deliberately gives an already-cycling page a fresh budget
rather than failing it on the first sweep after deploy.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "008_job_claim_attempts"
down_revision: str | None = "007_execution_target"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column(
            "claim_attempts",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("jobs", "claim_attempts")
