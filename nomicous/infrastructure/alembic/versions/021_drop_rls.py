"""remove row-level security and dedicated app database roles

Revision ID: 021_drop_rls
Revises: 020_public_project_read
Create Date: 2026-07-09

Application-layer authorization remains; Postgres RLS and split DB roles are
not used. Keeps query-performance indexes from 014.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "021_drop_rls"
down_revision: str | None = "020_public_project_read"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = (
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

_KNOWN_POLICIES = (
    ("users", "users_select_policy"),
    ("users", "users_insert_policy"),
    ("users", "users_update_policy"),
    ("projects", "projects_access_policy"),
    ("projects", "projects_select_policy"),
    ("projects", "projects_insert_policy"),
    ("projects", "projects_update_policy"),
    ("projects", "projects_delete_policy"),
    ("project_shared_users", "project_shared_users_access_policy"),
    ("inference_models", "inference_models_access_policy"),
    ("documents", "documents_access_policy"),
    ("document_parts", "document_parts_access_policy"),
    ("blocks", "blocks_access_policy"),
    ("lines", "lines_access_policy"),
    ("model_bindings", "model_bindings_access_policy"),
    ("jobs", "jobs_access_policy"),
    ("transcriptions", "transcriptions_access_policy"),
    ("line_transcriptions", "line_transcriptions_access_policy"),
    ("page_transcription_lines", "page_transcription_lines_access_policy"),
    ("annotation_history_snapshots", "annotation_history_snapshots_access_policy"),
    ("auth_rate_limit_attempts", "auth_rate_limit_attempts_access_policy"),
)


def upgrade() -> None:
    for table, policy in _KNOWN_POLICIES:
        op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")

    for table in _TABLES:
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
        REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM inference_worker;
        REVOKE ALL ON SCHEMA public FROM inference_worker;
        REVOKE ALL ON ALL TABLES IN SCHEMA public FROM nomicous_app;
        REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM nomicous_app;
        REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM nomicous_app;
        REVOKE ALL ON SCHEMA public FROM nomicous_app;

        ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
          REVOKE ALL ON TABLES FROM nomicous_app;
        ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
          REVOKE ALL ON SEQUENCES FROM nomicous_app;
        ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
          REVOKE ALL ON TABLES FROM inference_worker;
        ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
          REVOKE ALL ON SEQUENCES FROM inference_worker;

        GRANT ALL ON SCHEMA public TO PUBLIC;

        DROP ROLE IF EXISTS inference_worker;
        DROP ROLE IF EXISTS nomicous_app;
        """
    )

    op.create_index("ix_jobs_user_id", "jobs", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_jobs_user_id", table_name="jobs")
    # RLS re-enable is intentionally not automated; restore from 015–020 if needed.
