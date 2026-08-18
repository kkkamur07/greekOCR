"""Create helper_devices and helper_pairings for outbound device pairing.

The inference helper does not listen on loopback; it authenticates outbound with
an opaque device token. These two tables hold the credential *hashes* and the
short-lived pairing requests that mint them. No raw secret is ever stored.

``helper_devices.inference_host`` is folded in from what used to be
``007_execution_target``: a hosted worker is a device like any other (ADR 0003),
so capacity is one query over one table instead of a device check plus a separate
notion of cloud uptime. ``local`` is the right default - every device paired
before the column existed was a researcher's own computer paired from a browser.

The ``_has_table`` guards the pre-squash version carried are gone. They existed
for databases stamped while 001 was still regenerated from ORM metadata, and no
such database is left; a partial replay would now fail in 001's unguarded
``CREATE TABLE`` long before reaching this revision anyway.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003_helper_devices"
down_revision: str | None = "002_service_roles"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Created by 001; this revision only references it.
EXECUTION_TARGET = postgresql.ENUM("local", "cloud", name="execution_target", create_type=False)


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
        # Which host a paired device *is*. Last, matching the column order the
        # nine-revision chain produced when 007 added it with ALTER TABLE.
        sa.Column(
            "inference_host",
            EXECUTION_TARGET,
            server_default="local",
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
    # Capacity reads this too, rather than growing a second liveness mechanism.
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
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomikos_api') THEN
            GRANT SELECT, INSERT, UPDATE, DELETE
              ON TABLE helper_devices, helper_pairings TO nomikos_api;
          END IF;
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomikos_migrator') THEN
            GRANT ALL PRIVILEGES
              ON TABLE helper_devices, helper_pairings TO nomikos_migrator;
          END IF;
        END
        $$;
        """
    )


def upgrade() -> None:
    _create_helper_devices()
    _create_helper_pairings()
    _grant_runtime_privileges()


def downgrade() -> None:
    # helper_pairings holds the FK to helper_devices, so it goes first.
    op.drop_index("ix_helper_pairings_expires_at", table_name="helper_pairings")
    op.drop_table("helper_pairings")
    op.drop_index("ix_helper_devices_last_seen_at", table_name="helper_devices")
    op.drop_index("ix_helper_devices_user_live", table_name="helper_devices")
    op.drop_table("helper_devices")
