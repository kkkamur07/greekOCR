"""allow INSERT RETURNING under RLS for owned projects

Revision ID: 018_rls_insert_returning_select
Revises: 017_inference_worker_truncate
Create Date: 2026-07-09

PostgreSQL evaluates SELECT policies for INSERT ... RETURNING. A policy that
only checks app_user_can_access_project(id) cannot see the row being inserted
in the same statement, so ORM inserts fail. Include a direct owner_id check.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "018_rls_insert_returning_select"
down_revision: str | None = "017_inference_worker_truncate"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP POLICY IF EXISTS projects_select_policy ON projects")
    op.execute(
        """
        CREATE POLICY projects_select_policy ON projects
          FOR SELECT
          USING (
            app_rls_bypass()
            OR owner_id = app_current_user_id()
            OR app_user_can_access_project(id)
          );
        """
    )

    op.execute("DROP POLICY IF EXISTS documents_access_policy ON documents")
    op.execute(
        """
        CREATE POLICY documents_access_policy ON documents
          FOR ALL
          USING (
            app_rls_bypass()
            OR app_user_can_access_project(project_id)
            OR app_user_can_access_document(id)
          )
          WITH CHECK (
            app_rls_bypass()
            OR app_user_can_access_project(project_id)
          );
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS projects_select_policy ON projects")
    op.execute(
        """
        CREATE POLICY projects_select_policy ON projects
          FOR SELECT
          USING (app_rls_bypass() OR app_user_can_access_project(id));
        """
    )

    op.execute("DROP POLICY IF EXISTS documents_access_policy ON documents")
    op.execute(
        """
        CREATE POLICY documents_access_policy ON documents
          FOR ALL
          USING (app_user_can_access_document(id))
          WITH CHECK (app_rls_bypass() OR app_user_can_access_project(project_id));
        """
    )
