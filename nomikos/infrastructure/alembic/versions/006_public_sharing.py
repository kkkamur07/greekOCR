"""Add a revocable share token to documents and a per-part publish flag.

Before this revision, publishing a document (``workflow = 'published'``) made it and
every one of its parts readable by anyone who guessed or was handed the two path UUIDs
- there was no secret in the URL, so the only way to take a document back offline was to
unpublish it, which also broke every legitimate link that had ever been shared.

``documents.public_share_token`` is that secret. It is minted the first time a document
is published (see ``DocumentCatalog.update_document``) and can be rotated on demand,
which invalidates every link built from the old value without touching ``workflow``.
Nullable and unique: a document that has never been published has no token to leak, and
two documents can never collide on one.

``document_parts.published`` lets a chapter go live with some pages held back. Every
row gets ``true`` here - both the column default and this migration's backfill - so nothing
already reachable through a published document goes dark the moment this ships.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "006_public_sharing"
down_revision: str | None = "005_case_insensitive_email"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("public_share_token", sa.String(length=64), nullable=True),
    )
    op.create_index(
        op.f("ix_documents_public_share_token"),
        "documents",
        ["public_share_token"],
        unique=True,
    )
    op.add_column(
        "document_parts",
        sa.Column("published", sa.Boolean(), server_default="true", nullable=False),
    )


# Downgrading drops ``document_parts.published`` outright, and a later re-upgrade
# brings every part back at the ``true`` server default. A page an owner deliberately
# held back therefore becomes publicly reachable again, silently. Losing the token on
# the way down fails closed - every existing link simply 404s - but this one fails
# open, so a downgrade on a database with held-back pages needs those pages recorded
# first and set back afterwards.
def downgrade() -> None:
    op.drop_column("document_parts", "published")
    op.drop_index(op.f("ix_documents_public_share_token"), table_name="documents")
    op.drop_column("documents", "public_share_token")
