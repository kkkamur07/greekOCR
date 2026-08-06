"""Index the agent claim, which is the hottest query in the schema.

``job_claim_service._claimable_job_query`` filters ``status``, ``type``,
``execution_target`` and - for a device token - ``user_id``, then orders by
``created_at, id``. The only index that touched it was
``ix_jobs_claim_pending (created_at, id) WHERE status = 'pending'``, which serves
the *platform worker's* claim and answers none of the two selective predicates
the agent adds. So every claim walked the pending queue oldest-first, discarding
rows belonging to another **inference host** or another account until it found
one it could take.

That is a sequential-ish scan per claim, and every connected agent issues one per
second for as long as it is idle - the long poll re-checks on
``DEVICE_CLAIM_POLL_INTERVAL_SECONDS``. The cost grows with the depth of the
queue, which is exactly when it must not: a researcher's laptop looking for its
own page gets slower the more cloud work is queued ahead of it.

The new index leads with the two equality predicates and ends with the sort key,
so one range scan answers the filter and the ordering together and Postgres never
sorts. ``type`` is deliberately not in it: ``AGENT_CLAIMED_JOB_TYPES`` is an
``IN`` over two of the four values, which is not selective enough to earn a
column ahead of the sort key.

Partial on ``status = 'pending'`` for the same reason the existing index is: a
claimable row is pending by definition, and terminal rows are the overwhelming
majority of the table on any database that has been used.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "009_jobs_claim_target_index"
down_revision: str | None = "008_job_claim_attempts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "ix_jobs_claim_target_pending"


def upgrade() -> None:
    op.create_index(
        _INDEX_NAME,
        "jobs",
        ["execution_target", "user_id", "created_at", "id"],
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="jobs")
