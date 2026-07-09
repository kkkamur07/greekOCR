"""allow public read access to projects with published documents

Revision ID: 020_public_project_read
Revises: 019_users_directory_select
Create Date: 2026-07-09

Anonymous public routes load the parent project before returning a published
document. Extend app_user_can_access_project so app.public_read sessions can
see projects that contain at least one published document.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "020_public_project_read"
down_revision: str | None = "019_users_directory_select"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app_user_can_access_project(project_uuid uuid) RETURNS boolean
        LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
          SELECT CASE
            WHEN app_rls_bypass() THEN true
            WHEN app_public_read_enabled() AND EXISTS (
              SELECT 1
              FROM documents d
              WHERE d.project_id = project_uuid
                AND d.workflow = 'published'
            ) THEN true
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
        """
    )


def downgrade() -> None:
    op.execute(
        """
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
        """
    )
