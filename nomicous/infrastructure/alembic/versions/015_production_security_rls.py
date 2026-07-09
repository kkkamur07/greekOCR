"""production security — database roles and row-level security

Revision ID: 015_production_security_rls
Revises: 014_query_performance_indexes
Create Date: 2026-07-09

"""

from collections.abc import Sequence

from alembic import op

revision: str = "015_production_security_rls"
down_revision: str | None = "014_query_performance_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APP_TABLES = (
    "users",
    "projects",
    "project_shared_users",
    "inference_models",
    "documents",
    "document_parts",
    "blocks",
    "lines",
    "model_bindings",
    "jobs",
    "transcriptions",
    "line_transcriptions",
    "page_transcription_lines",
    "annotation_history_snapshots",
    "auth_rate_limit_attempts",
)


def upgrade() -> None:
  op.execute(
      """
      CREATE OR REPLACE FUNCTION app_rls_bypass() RETURNS boolean
      LANGUAGE sql STABLE AS $$
        SELECT coalesce(nullif(current_setting('app.bypass_rls', true), ''), 'false') = 'true';
      $$;

      CREATE OR REPLACE FUNCTION app_current_user_id() RETURNS uuid
      LANGUAGE sql STABLE AS $$
        SELECT nullif(current_setting('app.current_user_id', true), '')::uuid;
      $$;

      CREATE OR REPLACE FUNCTION app_public_read_enabled() RETURNS boolean
      LANGUAGE sql STABLE AS $$
        SELECT coalesce(nullif(current_setting('app.public_read', true), ''), 'false') = 'true';
      $$;

      CREATE OR REPLACE FUNCTION app_auth_lookup_enabled() RETURNS boolean
      LANGUAGE sql STABLE AS $$
        SELECT coalesce(nullif(current_setting('app.auth_lookup', true), ''), 'false') = 'true';
      $$;

      CREATE OR REPLACE FUNCTION app_user_can_access_project(project_uuid uuid) RETURNS boolean
      LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
        SELECT CASE
          WHEN app_rls_bypass() THEN true
          WHEN app_current_user_id() IS NULL THEN false
          ELSE EXISTS (
            SELECT 1
            FROM projects p
            WHERE p.id = project_uuid
              AND (
                p.owner_id = app_current_user_id()
                OR EXISTS (
                  SELECT 1
                  FROM project_shared_users psu
                  WHERE psu.project_id = p.id
                    AND psu.user_id = app_current_user_id()
                )
              )
          )
        END;
      $$;

      CREATE OR REPLACE FUNCTION app_user_can_access_document(document_uuid uuid) RETURNS boolean
      LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
        SELECT CASE
          WHEN app_rls_bypass() THEN true
          WHEN app_public_read_enabled() AND EXISTS (
            SELECT 1
            FROM documents d
            WHERE d.id = document_uuid
              AND d.workflow = 'published'
          ) THEN true
          WHEN app_current_user_id() IS NULL THEN false
          ELSE EXISTS (
            SELECT 1
            FROM documents d
            WHERE d.id = document_uuid
              AND app_user_can_access_project(d.project_id)
          )
        END;
      $$;

      CREATE OR REPLACE FUNCTION app_user_can_access_part(part_uuid uuid) RETURNS boolean
      LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
        SELECT CASE
          WHEN app_rls_bypass() THEN true
          WHEN app_current_user_id() IS NULL AND NOT app_public_read_enabled() THEN false
          ELSE EXISTS (
            SELECT 1
            FROM document_parts dp
            WHERE dp.id = part_uuid
              AND app_user_can_access_document(dp.document_id)
          )
        END;
      $$;

      CREATE OR REPLACE FUNCTION app_user_can_access_binding(
        binding_project_id uuid,
        binding_document_id uuid,
        binding_document_part_id uuid
      ) RETURNS boolean
      LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
        SELECT CASE
          WHEN app_rls_bypass() THEN true
          WHEN binding_document_part_id IS NOT NULL THEN app_user_can_access_part(binding_document_part_id)
          WHEN binding_document_id IS NOT NULL THEN app_user_can_access_document(binding_document_id)
          WHEN binding_project_id IS NOT NULL THEN app_user_can_access_project(binding_project_id)
          ELSE app_current_user_id() IS NOT NULL
        END;
      $$;
      """
  )

  op.execute(
      """
      DO $$
      BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nomicous_app') THEN
          CREATE ROLE nomicous_app LOGIN PASSWORD 'dev';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'inference_worker') THEN
          CREATE ROLE inference_worker LOGIN PASSWORD 'dev';
        END IF;
      END
      $$;

      REVOKE ALL ON SCHEMA public FROM PUBLIC;
      GRANT USAGE ON SCHEMA public TO nomicous_app;
      GRANT USAGE ON SCHEMA public TO inference_worker;

      GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nomicous_app;
      GRANT TRUNCATE ON ALL TABLES IN SCHEMA public TO nomicous_app;
      GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nomicous_app;

      REVOKE ALL ON ALL TABLES IN SCHEMA public FROM inference_worker;
      GRANT SELECT, INSERT, UPDATE, DELETE ON inference_jobs TO inference_worker;
      GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO inference_worker;

      ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON TABLES TO nomicous_app;
      ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT USAGE, SELECT ON SEQUENCES TO nomicous_app;
      """
  )

  for table in _APP_TABLES:
      op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
      op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

  op.execute(
      """
      CREATE POLICY users_select_policy ON users
        FOR SELECT
        USING (
          app_rls_bypass()
          OR app_auth_lookup_enabled()
          OR id = app_current_user_id()
        );

      CREATE POLICY users_insert_policy ON users
        FOR INSERT
        WITH CHECK (app_rls_bypass() OR app_auth_lookup_enabled());

      CREATE POLICY users_update_policy ON users
        FOR UPDATE
        USING (app_rls_bypass() OR id = app_current_user_id())
        WITH CHECK (app_rls_bypass() OR id = app_current_user_id());

      CREATE POLICY projects_access_policy ON projects
        FOR ALL
        USING (app_rls_bypass() OR app_user_can_access_project(id))
        WITH CHECK (
          app_rls_bypass()
          OR owner_id = app_current_user_id()
          OR app_user_can_access_project(id)
        );

      CREATE POLICY project_shared_users_access_policy ON project_shared_users
        FOR ALL
        USING (app_rls_bypass() OR app_user_can_access_project(project_id))
        WITH CHECK (app_rls_bypass() OR app_user_can_access_project(project_id));

      CREATE POLICY inference_models_access_policy ON inference_models
        FOR ALL
        USING (app_rls_bypass() OR app_current_user_id() IS NOT NULL)
        WITH CHECK (app_rls_bypass() OR app_current_user_id() IS NOT NULL);

      CREATE POLICY documents_access_policy ON documents
        FOR ALL
        USING (app_user_can_access_document(id))
        WITH CHECK (app_rls_bypass() OR app_user_can_access_project(project_id));

      CREATE POLICY document_parts_access_policy ON document_parts
        FOR ALL
        USING (app_user_can_access_document(document_id))
        WITH CHECK (app_user_can_access_document(document_id));

      CREATE POLICY blocks_access_policy ON blocks
        FOR ALL
        USING (
          EXISTS (
            SELECT 1
            FROM document_parts dp
            WHERE dp.id = blocks.part_id
              AND app_user_can_access_document(dp.document_id)
          )
        )
        WITH CHECK (
          EXISTS (
            SELECT 1
            FROM document_parts dp
            WHERE dp.id = blocks.part_id
              AND app_user_can_access_document(dp.document_id)
          )
        );

      CREATE POLICY lines_access_policy ON lines
        FOR ALL
        USING (app_user_can_access_part(part_id))
        WITH CHECK (app_user_can_access_part(part_id));

      CREATE POLICY model_bindings_access_policy ON model_bindings
        FOR ALL
        USING (
          app_user_can_access_binding(project_id, document_id, document_part_id)
        )
        WITH CHECK (
          app_user_can_access_binding(project_id, document_id, document_part_id)
        );

      CREATE POLICY jobs_access_policy ON jobs
        FOR ALL
        USING (
          app_rls_bypass()
          OR user_id = app_current_user_id()
          OR (document_id IS NOT NULL AND app_user_can_access_document(document_id))
        )
        WITH CHECK (
          app_rls_bypass()
          OR user_id = app_current_user_id()
          OR (document_id IS NOT NULL AND app_user_can_access_document(document_id))
        );

      CREATE POLICY transcriptions_access_policy ON transcriptions
        FOR ALL
        USING (app_user_can_access_document(document_id))
        WITH CHECK (app_user_can_access_document(document_id));

      CREATE POLICY line_transcriptions_access_policy ON line_transcriptions
        FOR ALL
        USING (
          EXISTS (
            SELECT 1
            FROM lines l
            WHERE l.id = line_transcriptions.line_id
              AND app_user_can_access_part(l.part_id)
          )
        )
        WITH CHECK (
          EXISTS (
            SELECT 1
            FROM lines l
            WHERE l.id = line_transcriptions.line_id
              AND app_user_can_access_part(l.part_id)
          )
        );

      CREATE POLICY page_transcription_lines_access_policy ON page_transcription_lines
        FOR ALL
        USING (app_user_can_access_part(part_id))
        WITH CHECK (app_user_can_access_part(part_id));

      CREATE POLICY annotation_history_snapshots_access_policy ON annotation_history_snapshots
        FOR ALL
        USING (app_user_can_access_part(part_id))
        WITH CHECK (app_user_can_access_part(part_id));

      CREATE POLICY auth_rate_limit_attempts_access_policy ON auth_rate_limit_attempts
        FOR ALL
        USING (app_rls_bypass())
        WITH CHECK (app_rls_bypass());
      """
  )


def downgrade() -> None:
  policies = (
      "users_select_policy",
      "users_insert_policy",
      "users_update_policy",
      "projects_access_policy",
      "project_shared_users_access_policy",
      "inference_models_access_policy",
      "documents_access_policy",
      "document_parts_access_policy",
      "blocks_access_policy",
      "lines_access_policy",
      "model_bindings_access_policy",
      "jobs_access_policy",
      "transcriptions_access_policy",
      "line_transcriptions_access_policy",
      "page_transcription_lines_access_policy",
      "annotation_history_snapshots_access_policy",
      "auth_rate_limit_attempts_access_policy",
  )
  table_by_policy = {
      "users_select_policy": "users",
      "users_insert_policy": "users",
      "users_update_policy": "users",
      "projects_access_policy": "projects",
      "project_shared_users_access_policy": "project_shared_users",
      "inference_models_access_policy": "inference_models",
      "documents_access_policy": "documents",
      "document_parts_access_policy": "document_parts",
      "blocks_access_policy": "blocks",
      "lines_access_policy": "lines",
      "model_bindings_access_policy": "model_bindings",
      "jobs_access_policy": "jobs",
      "transcriptions_access_policy": "transcriptions",
      "line_transcriptions_access_policy": "line_transcriptions",
      "page_transcription_lines_access_policy": "page_transcription_lines",
      "annotation_history_snapshots_access_policy": "annotation_history_snapshots",
      "auth_rate_limit_attempts_access_policy": "auth_rate_limit_attempts",
  }
  for policy in policies:
      op.execute(f"DROP POLICY IF EXISTS {policy} ON {table_by_policy[policy]}")

  for table in _APP_TABLES:
      op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
      op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

  op.execute(
      """
      DROP FUNCTION IF EXISTS app_user_can_access_binding(uuid, uuid, uuid);
      DROP FUNCTION IF EXISTS app_user_can_access_part(uuid);
      DROP FUNCTION IF EXISTS app_user_can_access_document(uuid);
      DROP FUNCTION IF EXISTS app_user_can_access_project(uuid);
      DROP FUNCTION IF EXISTS app_auth_lookup_enabled();
      DROP FUNCTION IF EXISTS app_public_read_enabled();
      DROP FUNCTION IF EXISTS app_current_user_id();
      DROP FUNCTION IF EXISTS app_rls_bypass();

      REVOKE ALL ON ALL TABLES IN SCHEMA public FROM inference_worker;
      REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM inference_worker;
      REVOKE ALL ON ALL TABLES IN SCHEMA public FROM nomicous_app;
      REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM nomicous_app;
      GRANT ALL ON SCHEMA public TO PUBLIC;

      DROP ROLE IF EXISTS inference_worker;
      DROP ROLE IF EXISTS nomicous_app;
      """
  )
