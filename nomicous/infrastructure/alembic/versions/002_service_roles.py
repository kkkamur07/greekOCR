"""Create least-privilege service role groups and grants.

Provider-managed LOGIN principals and their memberships remain outside this
migration. On providers that do not grant CREATEROLE to the migrator, the
operator should run scripts/platform/provision_database_roles.sql after the
schema migration.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "002_service_roles"
down_revision: str | None = "001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
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
                EXECUTE format(
                  'CREATE ROLE %I NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION',
                  role_name
                );
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
              'Service role groups were not created; run the provider role bootstrap separately.';
            RETURN;
          END IF;

          REVOKE ALL ON SCHEMA public FROM PUBLIC;
          REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
          REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC;

          GRANT USAGE, CREATE ON SCHEMA public TO nomicous_migrator;
          GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nomicous_migrator;
          GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nomicous_migrator;

          GRANT USAGE ON SCHEMA public TO nomicous_api;
          GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nomicous_api;
          GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nomicous_api;

          GRANT USAGE ON SCHEMA public TO nomicous_platform_worker;
          GRANT SELECT, UPDATE ON TABLE jobs TO nomicous_platform_worker;
          GRANT SELECT ON TABLE documents, document_parts, blocks, lines,
            model_bindings, inference_models TO nomicous_platform_worker;

          GRANT USAGE ON SCHEMA public TO nomicous_inference_worker;
          GRANT SELECT, UPDATE ON TABLE inference_jobs TO nomicous_inference_worker;

          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT ALL PRIVILEGES ON TABLES TO nomicous_migrator;
          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT ALL PRIVILEGES ON SEQUENCES TO nomicous_migrator;

          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO nomicous_api;
          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT USAGE, SELECT ON SEQUENCES TO nomicous_api;

          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            REVOKE ALL ON TABLES FROM PUBLIC;
          ALTER DEFAULT PRIVILEGES IN SCHEMA public
            REVOKE ALL ON SEQUENCES FROM PUBLIC;
        END
        $$;
        """
    )


def downgrade() -> None:
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
              EXECUTE format('REVOKE ALL ON SCHEMA public FROM %I', role_name);
            END IF;
          END LOOP;
        END
        $$;
        """
    )
