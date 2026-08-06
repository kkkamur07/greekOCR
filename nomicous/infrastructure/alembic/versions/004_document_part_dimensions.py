"""Guarantee document_parts.width/height exist for persisted page dimensions.

The upload path now records the page dimensions from the decode it already performs, so
``document_parts.width``/``height`` stop being write-only ORM columns.

This is a real change on every database, including a fresh one. It did not used to be:
``001_initial_schema`` was regenerated from live ORM metadata, so a database created from
scratch already had both columns and this revision was a no-op there. The squash froze 001
to stop that, and 001 now names ``document_parts.width`` / ``height`` as deliberately
absent from the baseline.

The presence check stays, and its only remaining subject is a database stamped during the
period 001 was still being regenerated: those already have the columns and must not fail
on a second ``ADD COLUMN``. On any database created since the freeze, it never fires.

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
    """Drop what ``upgrade`` added.

    This used to be a no-op on the grounds that "the columns belong to the 001
    baseline schema and are dropped with it". That was true only while 001 was
    regenerated from ORM metadata. Since the squash froze it, 001 does not create
    these columns, so a no-op here left the chain irreversible: downgrading to
    002 and upgrading again is fine, but downgrading to 004 and inspecting the
    schema showed columns that revision is supposed to own.

    ``IF EXISTS`` because a database stamped before the freeze got them from 001
    and may have had them dropped by something else first.
    """
    for column in _COLUMNS:
        op.execute(f"ALTER TABLE document_parts DROP COLUMN IF EXISTS {column}")
