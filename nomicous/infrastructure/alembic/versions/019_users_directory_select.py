"""allow authenticated user directory lookups for sharing

Revision ID: 019_users_directory_select
Revises: 018_rls_insert_returning_select
Create Date: 2026-07-09

Project sharing resolves collaborators by username. Authenticated members need
read access to other users' directory rows without opening auth registration.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "019_users_directory_select"
down_revision: str | None = "018_rls_insert_returning_select"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP POLICY IF EXISTS users_select_policy ON users")
    op.execute(
        """
        CREATE POLICY users_select_policy ON users
          FOR SELECT
          USING (
            app_rls_bypass()
            OR app_auth_lookup_enabled()
            OR id = app_current_user_id()
            OR app_current_user_id() IS NOT NULL
          );
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS users_select_policy ON users")
    op.execute(
        """
        CREATE POLICY users_select_policy ON users
          FOR SELECT
          USING (
            app_rls_bypass()
            OR app_auth_lookup_enabled()
            OR id = app_current_user_id()
          );
        """
    )
