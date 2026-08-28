"""Rename nomicous_* service roles to nomikos_* for the project rename.

001-003 were edited in place to say ``nomikos_*``, which is correct for a
fresh database, but does nothing for one that already ran the chain under the
old names: Alembic never replays an applied revision, so such a database still
holds ``nomicous_*`` roles while every consumer now asks for ``nomikos_*``.
This revision closes that gap.

``ALTER ROLE ... RENAME TO`` carries everything that matters with it: grants,
default privileges, and memberships all follow the role's OID, so nothing from
002 needs to be re-granted. These are NOLOGIN group roles with no passwords,
so the MD5-password-cleared-on-rename caveat does not apply.

Each rename is guarded twice: it runs only when the old name exists and the
new one does not, so a fresh database and one already renamed both pass
through as a no-op. Like 002, the whole block is skipped when the migrating
user cannot manage roles; the provider role bootstrap owns the rename then.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "004_rename_service_roles"
down_revision: str | None = "003_helper_devices"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        DECLARE
          can_manage_roles boolean;
          suffix text;
        BEGIN
          SELECT rolsuper OR rolcreaterole
          INTO can_manage_roles
          FROM pg_roles
          WHERE rolname = current_user;

          IF NOT coalesce(can_manage_roles, false) THEN
            RAISE NOTICE
              'Service roles were not renamed; run the provider role bootstrap separately.';
            RETURN;
          END IF;

          FOREACH suffix IN ARRAY ARRAY['migrator', 'api', 'platform_worker', 'inference_worker']
          LOOP
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomicous_' || suffix)
              AND NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomikos_' || suffix)
            THEN
              EXECUTE format(
                'ALTER ROLE %I RENAME TO %I',
                'nomicous_' || suffix,
                'nomikos_' || suffix
              );
            END IF;
          END LOOP;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        DECLARE
          can_manage_roles boolean;
          suffix text;
        BEGIN
          SELECT rolsuper OR rolcreaterole
          INTO can_manage_roles
          FROM pg_roles
          WHERE rolname = current_user;

          IF NOT coalesce(can_manage_roles, false) THEN
            RETURN;
          END IF;

          FOREACH suffix IN ARRAY ARRAY['migrator', 'api', 'platform_worker', 'inference_worker']
          LOOP
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomikos_' || suffix)
              AND NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomicous_' || suffix)
            THEN
              EXECUTE format(
                'ALTER ROLE %I RENAME TO %I',
                'nomikos_' || suffix,
                'nomicous_' || suffix
              );
            END IF;
          END LOOP;
        END
        $$;
        """
    )
