"""grant inference worker truncate on inference_jobs

Revision ID: 017_inference_worker_truncate
Revises: 016_project_rls_insert_fix
Create Date: 2026-07-09

"""

from collections.abc import Sequence

from alembic import op

revision: str = "017_inference_worker_truncate"
down_revision: str | None = "016_project_rls_insert_fix"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("GRANT TRUNCATE ON inference_jobs TO inference_worker")


def downgrade() -> None:
    op.execute("REVOKE TRUNCATE ON inference_jobs FROM inference_worker")
