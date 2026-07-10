"""Create least-privilege database group roles for service processes.

Revision ID: 025_service_roles
Revises: 024_browser_sessions
Create Date: 2026-07-10

The roles are NOLOGIN groups. Operators create or select provider-managed LOGIN
principals, grant each one exactly one group role, and keep credentials in their
secret manager. The migration deliberately keeps the current schema owner and
does not set role passwords, so existing local and Supabase operator workflows
continue to work.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "025_service_roles"
down_revision: str | None = "024_browser_sessions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ROLES = (
    "nomicous_migrator",
    "nomicous_api",
    "nomicous_platform_worker",
    "nomicous_inference_worker",
)


def upgrade() -> None:
    # Managed Postgres providers can restrict CREATEROLE. In that case the
    # current operator remains usable and the runbook supplies the one-time
    # privileged bootstrap instead of making an otherwise-safe schema upgrade
    # fail.
    op.execute(
        """
        DO $$
        DECLARE
          can_create_roles boolean;
          role_name text;
          roles_ready boolean;
        BEGIN
          SELECT rolsuper OR rolcreaterole
          INTO can_create_roles
          FROM pg_roles
          WHERE rolname = current_user;

          IF coalesce(can_create_roles, false) THEN
            FOREACH role_name IN ARRAY ARRAY[
              'nomicous_migrator',
              'nomicous_api',
              'nomicous_platform_worker',
              'nomicous_inference_worker'
            ]
            LOOP
              IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
                EXECUTE format('CREATE ROLE %I NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION', role_name);
              END IF;
            END LOOP;
          END IF;

          SELECT bool_and(EXISTS (
            SELECT 1 FROM pg_roles WHERE rolname = required_role
          ))
          INTO roles_ready
          FROM unnest(ARRAY[
            'nomicous_migrator',
            'nomicous_api',
            'nomicous_platform_worker',
            'nomicous_inference_worker'
          ]) AS required_role;

          IF NOT coalesce(roles_ready, false) THEN
            RAISE NOTICE
              'Nomicous service roles were not created. Run the database-role bootstrap with a CREATEROLE-capable provider operator, then grant the roles to managed login principals.';
            RETURN;
          END IF;

          REVOKE ALL ON SCHEMA public FROM PUBLIC;
          REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
          REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC;

          GRANT USAGE, CREATE ON SCHEMA public TO nomicous_migrator;
          GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nomicous_migrator;
          GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nomicous_migrator;
          GRANT USAGE ON ALL TYPES IN SCHEMA public TO nomicous_migrator;

          GRANT USAGE ON SCHEMA public TO nomicous_api;
          GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nomicous_api;
          GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nomicous_api;
          GRANT USAGE ON ALL TYPES IN SCHEMA public TO nomicous_api;

          GRANT USAGE ON SCHEMA public TO nomicous_platform_worker;
          GRANT SELECT, UPDATE ON TABLE jobs TO nomicous_platform_worker;
          GRANT SELECT ON TABLE documents, document_parts, blocks, lines, model_bindings, inference_models
            TO nomicous_platform_worker;
          GRANT USAGE ON ALL TYPES IN SCHEMA public TO nomicous_platform_worker;

          GRANT USAGE ON SCHEMA public TO nomicous_inference_worker;
          GRANT SELECT, UPDATE ON TABLE inference_jobs TO nomicous_inference_worker;
          GRANT USAGE ON ALL TYPES IN SCHEMA public TO nomicous_inference_worker;

          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT ALL PRIVILEGES ON TABLES TO nomicous_migrator;
          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT ALL PRIVILEGES ON SEQUENCES TO nomicous_migrator;
          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT USAGE ON TYPES TO nomicous_migrator;

          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO nomicous_api;
          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT USAGE, SELECT ON SEQUENCES TO nomicous_api;
          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT USAGE ON TYPES TO nomicous_api;

          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            REVOKE ALL ON TABLES FROM PUBLIC;
          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            REVOKE ALL ON SEQUENCES FROM PUBLIC;
          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            REVOKE ALL ON TYPES FROM PUBLIC;
        END
        $$;
        """
    )


def downgrade() -> None:
    # Do not drop role groups: operators may have assigned provider-managed
    # login principals to them. Removing grants is reversible and non-disruptive.
    op.execute(
        """
        DO $$
        DECLARE
          role_name text;
        BEGIN
          FOREACH role_name IN ARRAY ARRAY[
            'nomicous_migrator',
            'nomicous_api',
            'nomicous_platform_worker',
            'nomicous_inference_worker'
          ]
          LOOP
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
              EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA public FROM %I', role_name);
              EXECUTE format('REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM %I', role_name);
              EXECUTE format('REVOKE ALL ON ALL TYPES IN SCHEMA public FROM %I', role_name);
              EXECUTE format('REVOKE ALL ON SCHEMA public FROM %I', role_name);
            END IF;
          END LOOP;
        END
        $$;
        """
    )
