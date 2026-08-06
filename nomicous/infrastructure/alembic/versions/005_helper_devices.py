"""Create helper_devices and helper_pairings for outbound device pairing.

The inference helper stops listening on loopback and instead authenticates
outbound with an opaque device token. These two tables hold the credential
*hashes* and the short-lived pairing requests that mint them. No raw secret is
ever stored.

This is a real change on every database, including a fresh one. It did not used
to be: ``001_initial_schema`` was regenerated from live ORM metadata, so once
these models were registered in ``infrastructure/models.py`` the baseline built
both tables and this revision was a no-op there. The squash froze 001 to stop
precisely that, and 001 now names ``helper_devices`` and ``helper_pairings`` as
deliberately absent from the baseline.

The ``_has_table`` guards stay, and their only remaining subject is a database
stamped during the period 001 was still being regenerated: those already have the
tables and must not fail on a second ``CREATE TABLE``. On any database created
since the freeze, the guards never fire. ``_grant_runtime_privileges`` is outside
them either way - a database that already had the tables still needs the grants.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "005_helper_devices"
down_revision: str | None = "004_document_part_dimensions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _create_helper_devices() -> None:
    op.create_table(
        "helper_devices",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("platform", sa.String(length=32), nullable=False),
        sa.Column("helper_version", sa.String(length=32), nullable=False),
        sa.Column(
            "capabilities",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        # '' marks approved-but-not-collected; it can never equal a 64-hex digest.
        sa.Column("token_hash", sa.String(length=64), server_default="", nullable=False),
        sa.Column("previous_token_hash", sa.String(length=64), nullable=True),
        sa.Column("previous_token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("token_prefix", sa.String(length=20), server_default="", nullable=False),
        sa.Column("token_issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paired_from_ip", sa.String(length=45), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_ip", sa.String(length=45), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_helper_devices_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_helper_devices")),
    )
    # No index on token_hash: authentication is a primary-key fetch (the token
    # carries its own device id) plus one constant-time compare, never a scan.
    op.create_index(
        "ix_helper_devices_user_live",
        "helper_devices",
        ["user_id"],
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    op.create_index("ix_helper_devices_last_seen_at", "helper_devices", ["last_seen_at"])


def _create_helper_pairings() -> None:
    op.create_table(
        "helper_pairings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("device_code_hash", sa.String(length=64), nullable=False),
        sa.Column("verification_token_hash", sa.String(length=64), nullable=False),
        sa.Column("requested_name", sa.String(length=120), nullable=False),
        sa.Column("requested_platform", sa.String(length=32), nullable=False),
        sa.Column("requested_helper_version", sa.String(length=32), nullable=False),
        sa.Column(
            "requested_capabilities",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("request_ip", sa.String(length=45), nullable=False),
        sa.Column("request_user_agent", sa.String(length=255), nullable=True),
        sa.Column("attempts", sa.SmallInteger(), server_default="0", nullable=False),
        sa.Column("last_polled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("poll_interval_seconds", sa.SmallInteger(), server_default="5", nullable=False),
        sa.Column("delivery_count", sa.SmallInteger(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("approved_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("denied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("device_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["approved_user_id"],
            ["users.id"],
            name=op.f("fk_helper_pairings_approved_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["device_id"],
            ["helper_devices.id"],
            name=op.f("fk_helper_pairings_device_id_helper_devices"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_helper_pairings")),
        # The browser holds no row id, so verification_token_hash is the one
        # value looked up *by* digest and must be unique.
        sa.UniqueConstraint(
            "verification_token_hash", name=op.f("uq_helper_pairings_verification_token_hash")
        ),
    )
    # Serves both the live-pairing count and the sweep of finished rows. There is
    # no index on request_ip because nothing queries by it.
    op.create_index("ix_helper_pairings_expires_at", "helper_pairings", ["expires_at"])


def _grant_runtime_privileges() -> None:
    """Match what 002_service_roles grants the API role on every other table.

    002 issues ``GRANT ... ON ALL TABLES``, which is point-in-time, plus
    ``ALTER DEFAULT PRIVILEGES``, which only applies to tables created by the
    role that ran 002. Neither is guaranteed to cover a table created here, so
    the grant is explicit - otherwise the API role can read the schema and not
    the rows, and every pairing request fails with a permission error at runtime.
    """
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomicous_api') THEN
            GRANT SELECT, INSERT, UPDATE, DELETE
              ON TABLE helper_devices, helper_pairings TO nomicous_api;
          END IF;
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomicous_migrator') THEN
            GRANT ALL PRIVILEGES
              ON TABLE helper_devices, helper_pairings TO nomicous_migrator;
          END IF;
        END
        $$;
        """
    )


def upgrade() -> None:
    if not _has_table("helper_devices"):
        _create_helper_devices()
    if not _has_table("helper_pairings"):
        _create_helper_pairings()
    # Outside the has_table guards: a database stamped while 001 still built these
    # tables has them already and would otherwise never be granted anything.
    _grant_runtime_privileges()


def downgrade() -> None:
    # helper_pairings holds the FK to helper_devices, so it goes first.
    if _has_table("helper_pairings"):
        op.drop_index("ix_helper_pairings_expires_at", table_name="helper_pairings")
        op.drop_table("helper_pairings")
    if _has_table("helper_devices"):
        op.drop_index("ix_helper_devices_last_seen_at", table_name="helper_devices")
        op.drop_index("ix_helper_devices_user_live", table_name="helper_devices")
        op.drop_table("helper_devices")
