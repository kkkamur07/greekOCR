"""Drop the inference service's own job queue.

There is one job queue now and the platform owns it (ADR 0003). ``inference_jobs``
was a mailbox between two processes we control: the platform worker wrote a row
over HTTP, an inference worker claimed it, and the result came back through the
callback the ``jobs`` table already understood. Deleting the mailbox leaves the
four-step path a paired device already takes.

Cloud inference was off when this landed, so there is nothing to migrate — no
in-flight row is being discarded, because none can exist.

``nomicous_inference_worker`` loses everything it could reach: dropping the table
takes its table grants with it, and the schema grant is revoked below. The role
itself is left in place. Provider-managed LOGIN principals are members of these
groups and live outside migrations (see ``002_service_roles``), so dropping the
group here could fail on exactly the deployments that need this migration most.
Operators may drop it once no principal is a member.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "006_drop_inference_jobs"
down_revision: str | None = "005_helper_devices"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INFERENCE_JOB_STATUS = postgresql.ENUM(
    "pending",
    "running",
    "done",
    "failed",
    name="inference_job_status",
    create_type=False,
)


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if _has_table("inference_jobs"):
        op.drop_table("inference_jobs")
    INFERENCE_JOB_STATUS.drop(op.get_bind(), checkfirst=True)
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomicous_inference_worker') THEN
            REVOKE ALL ON SCHEMA public FROM nomicous_inference_worker;
          END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    """Recreate the queue table as ``001_initial_schema`` built it.

    Restoring the table does not restore the second queue: the code that read
    and wrote it is gone. This exists so the chain is reversible on a disposable
    database, not as a rollback plan.
    """
    if _has_table("inference_jobs"):
        return

    bind = op.get_bind()
    inference_task = postgresql.ENUM(
        "segment",
        "transcribe",
        "binarize",
        name="inference_task",
        create_type=False,
    )
    INFERENCE_JOB_STATUS.create(bind, checkfirst=True)
    op.create_table(
        "inference_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("product_job_id", sa.UUID(), nullable=False),
        sa.Column("task", inference_task, nullable=False),
        sa.Column("registry_model_id", sa.Text(), nullable=False),
        sa.Column("registry_tag", sa.Text(), nullable=False),
        sa.Column("status", INFERENCE_JOB_STATUS, nullable=False),
        sa.Column("image_bytes", sa.LargeBinary(), nullable=False),
        sa.Column(
            "params", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False
        ),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_jobs")),
    )
    op.create_index(op.f("ix_inference_jobs_product_job_id"), "inference_jobs", ["product_job_id"])
    op.create_index(op.f("ix_inference_jobs_status"), "inference_jobs", ["status"])
    op.create_index(
        "ix_inference_jobs_claim_order",
        "inference_jobs",
        ["status", "created_at", "id"],
    )
