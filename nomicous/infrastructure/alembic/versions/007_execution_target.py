"""Execution target on jobs, inference host on devices, host preference on accounts.

Three columns and one enum type, which together make **execution target** a fact
about a job rather than a string in a JSON payload:

* ``jobs.execution_target`` / ``jobs.preferred_execution_target`` - where the job
  runs, and where the account setting asked for it to run. They differ exactly
  when the preferred **inference host** had no **capacity** at submission, which
  is the substitution the researcher has to be told about.
* ``helper_devices.inference_host`` - which host a paired device *is*. A hosted
  worker is a device like any other (ADR 0003), so capacity is one query over one
  table instead of a device check plus a separate notion of cloud uptime.
* ``users.prefer_local_inference`` - "use my computer when it is available",
  chosen once at the account level. There is no per-job column, deliberately.

The defaults are ``cloud`` for jobs and ``local`` for devices, and both are
correct for the rows that already exist: every job written before this migration
was dispatched to the hosted inference service, and every device paired before it
was a researcher's own computer paired from a browser.

``local_only`` is not a value of this enum and never was one in this database.
See ADR 0002 for why it is not coming back.

The trigger is the part worth reading. The ORM raises on any attempt to move a
job between hosts, but an application guard only binds statements that go through
that mapper - and the platform issues bulk ``UPDATE`` statements against ``jobs``
from the stale sweep and the callback path. "Never changed afterwards" is a
property the whole of ADR 0002 rests on, so it is also enforced where nothing can
route around it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "007_execution_target"
down_revision: str | None = "006_drop_inference_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_EXECUTION_TARGET = postgresql.ENUM(
    "local", "cloud", name="execution_target", create_type=False
)

_FIX_TRIGGER_FUNCTION = """
CREATE OR REPLACE FUNCTION jobs_execution_target_is_fixed() RETURNS trigger AS $$
BEGIN
    IF NEW.execution_target IS DISTINCT FROM OLD.execution_target THEN
        RAISE EXCEPTION
            'jobs.execution_target is fixed at submission (job %: % -> %)',
            OLD.id, OLD.execution_target, NEW.execution_target
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    bind = op.get_bind()
    _EXECUTION_TARGET.create(bind, checkfirst=True)

    op.add_column(
        "jobs",
        sa.Column(
            "execution_target",
            _EXECUTION_TARGET,
            server_default="cloud",
            nullable=False,
        ),
    )
    op.add_column(
        "jobs",
        sa.Column(
            "preferred_execution_target",
            _EXECUTION_TARGET,
            server_default="cloud",
            nullable=False,
        ),
    )
    op.add_column(
        "helper_devices",
        sa.Column(
            "inference_host",
            _EXECUTION_TARGET,
            server_default="local",
            nullable=False,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "prefer_local_inference",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )

    op.execute(_FIX_TRIGGER_FUNCTION)
    op.execute(
        "CREATE TRIGGER jobs_execution_target_is_fixed "
        "BEFORE UPDATE ON jobs FOR EACH ROW "
        "EXECUTE FUNCTION jobs_execution_target_is_fixed()"
    )

    # No new index. Capacity reads ``ix_helper_devices_last_seen_at``, which the
    # device layer already carries - a second liveness mechanism is exactly what
    # this work is meant to avoid.


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS jobs_execution_target_is_fixed ON jobs")
    op.execute("DROP FUNCTION IF EXISTS jobs_execution_target_is_fixed()")
    op.drop_column("users", "prefer_local_inference")
    op.drop_column("helper_devices", "inference_host")
    op.drop_column("jobs", "preferred_execution_target")
    op.drop_column("jobs", "execution_target")
    _EXECUTION_TARGET.drop(op.get_bind(), checkfirst=True)
