"""Enforce case-insensitive email uniqueness with a functional index.

001 gave ``users.email`` a plain unique btree index, which treats
``Victim@x.com`` and ``victim@x.com`` as distinct rows. ``get_by_email`` now
looks up with ``func.lower(...)``; without a matching functional index that
query cannot use the plain one and falls back to a sequential scan, and the
plain unique index still lets the two case variants both register.

Replace it with a unique index on ``lower(email)``: it enforces
case-insensitive uniqueness at the database and backs the lookup. The
schema-layer normalisation keeps new rows lowercase, so on clean data the
build is a no-op; a database carrying pre-existing case-variant duplicates
must dedupe them before this revision applies.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "005_case_insensitive_email"
down_revision: str | None = "004_rename_service_roles"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_users_email_lower",
        "users",
        [sa.text("lower(email)")],
        unique=True,
    )
    op.drop_index(op.f("ix_users_email"), table_name="users")


def downgrade() -> None:
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.drop_index("uq_users_email_lower", table_name="users")
