"""Guarantee document_parts.width/height exist for persisted page dimensions.

The upload path now records the page dimensions from the decode it already performs, so
``document_parts.width``/``height`` stop being write-only ORM columns. The columns are
part of the squashed 001 baseline (created from ORM metadata); this migration only makes
their presence explicit and idempotent for databases whose baseline predates them.

Existing rows are NOT backfilled here. The dimensions of an already-uploaded page live
only inside the stored object (Supabase Storage or the local media root), and a migration
must not reach into object storage. Instead the read paths tolerate NULL: the document
read routes decode each legacy image once, persist the recovered dimensions, and every
later read is served from Postgres — see
``PartServiceMixin.backfill_part_dimensions``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "004_document_part_dimensions"
down_revision: str | None = "003_job_lifecycle"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = ("width", "height")


def _existing_columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {column["name"] for column in inspector.get_columns("document_parts")}


def upgrade() -> None:
    present = _existing_columns()
    for column in _COLUMNS:
        if column not in present:
            op.add_column("document_parts", sa.Column(column, sa.Integer(), nullable=True))


def downgrade() -> None:
    """No-op: the columns belong to the 001 baseline schema and are dropped with it."""
