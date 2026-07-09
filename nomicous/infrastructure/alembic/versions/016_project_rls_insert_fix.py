"""tighten project RLS insert policy

Revision ID: 016_project_rls_insert_fix
Revises: 015_production_security_rls
Create Date: 2026-07-09

"""

from collections.abc import Sequence

from alembic import op

revision: str = "016_project_rls_insert_fix"
down_revision: str | None = "015_production_security_rls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP POLICY IF EXISTS projects_access_policy ON projects")
    op.execute(
        """
        CREATE POLICY projects_select_policy ON projects
          FOR SELECT
          USING (app_rls_bypass() OR app_user_can_access_project(id));

        CREATE POLICY projects_insert_policy ON projects
          FOR INSERT
          WITH CHECK (app_rls_bypass() OR owner_id = app_current_user_id());

        CREATE POLICY projects_update_policy ON projects
          FOR UPDATE
          USING (app_rls_bypass() OR app_user_can_access_project(id))
          WITH CHECK (app_rls_bypass() OR app_user_can_access_project(id));

        CREATE POLICY projects_delete_policy ON projects
          FOR DELETE
          USING (app_rls_bypass() OR owner_id = app_current_user_id());
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS projects_delete_policy ON projects")
    op.execute("DROP POLICY IF EXISTS projects_update_policy ON projects")
    op.execute("DROP POLICY IF EXISTS projects_insert_policy ON projects")
    op.execute("DROP POLICY IF EXISTS projects_select_policy ON projects")
    op.execute(
        """
        CREATE POLICY projects_access_policy ON projects
          FOR ALL
          USING (app_rls_bypass() OR app_user_can_access_project(id))
          WITH CHECK (
            app_rls_bypass()
            OR owner_id = app_current_user_id()
            OR app_user_can_access_project(id)
          );
        """
    )
