"""Project CRUD and sharing use cases."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import AccessDeniedError, ConflictError, NotFoundError
from backend.project.domain.access import has_owner, is_member, is_owner
from backend.project.infrastructure.orm_models import Project
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User
from backend.users.infrastructure.user_repository import UserRepository


class ProjectService:
    def __init__(
        self,
        project_repo: ProjectRepository | None = None,
        user_repo: UserRepository | None = None,
    ) -> None:
        self._projects = project_repo or ProjectRepository()
        self._users = user_repo or UserRepository()

    async def list_projects(
        self,
        session: AsyncSession,
        user: User,
        *,
        limit: int = 50,
        cursor=None,
    ) -> list[Project]:
        return await self._projects.list_for_user(session, user.id, limit=limit, cursor=cursor)

    async def create_project(
        self,
        session: AsyncSession,
        user: User,
        *,
        name: str,
        slug: str,
        guidelines: str | None = None,
    ) -> Project:
        if await self._projects.get_by_slug(session, slug):
            raise ConflictError("Project slug already taken")
        return await self._projects.create(
            session,
            name=name,
            slug=slug,
            guidelines=guidelines,
            owner_id=user.id,
        )

    async def get_project(self, session: AsyncSession, user: User, project_id: UUID) -> Project:
        project = await self._load_or_404(session, project_id)
        self._require_member(project, user.id)
        return project

    async def update_project(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        **updates: str | None,
    ) -> Project:
        project = await self._load_or_404(session, project_id)
        self._require_owner(project, user.id)
        slug = updates.get("slug")
        if slug is not None and slug != project.slug:
            existing = await self._projects.get_by_slug(session, slug)
            if existing is not None:
                raise ConflictError("Project slug already taken")
        return await self._projects.update(session, project, **updates)

    async def delete_project(self, session: AsyncSession, user: User, project_id: UUID) -> None:
        project = await self._load_or_404(session, project_id)
        self._require_owner(project, user.id)
        await self._projects.delete(session, project)

    async def list_collaborators(
        self, session: AsyncSession, user: User, project_id: UUID
    ) -> list[User]:
        """The users a project is shared with. Owner-only: it exposes emails."""
        project = await self._load_or_404(session, project_id)
        self._require_owner(project, user.id)
        return sorted(project.shared_users, key=lambda shared: shared.username.lower())

    async def share_project(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        *,
        username: str | None = None,
        email: str | None = None,
    ) -> None:
        project = await self._load_or_404(session, project_id)
        self._require_owner(project, user.id)
        collaborator = await self._find_collaborator(session, username=username, email=email)
        if collaborator.id == project.owner_id:
            raise ConflictError("Cannot share project with the owner")
        if any(shared.id == collaborator.id for shared in project.shared_users):
            raise ConflictError("User already has access to this project")
        await self._projects.add_shared_user(session, project, collaborator)

    async def unshare_project(
        self, session: AsyncSession, user: User, project_id: UUID, *, username: str
    ) -> None:
        project = await self._load_or_404(session, project_id)
        self._require_owner(project, user.id)
        collaborator = await self._users.get_by_username(session, username)
        if collaborator is None:
            raise NotFoundError("User not found")
        removed = await self._projects.remove_shared_user(session, project, collaborator)
        if not removed:
            raise NotFoundError("User is not shared on this project")

    async def _find_collaborator(
        self, session: AsyncSession, *, username: str | None, email: str | None
    ) -> User:
        if (username is None) == (email is None):
            raise ValueError("share_project needs exactly one of username or email")
        if email is not None:
            collaborator = await self._users.get_by_email(session, email)
            if collaborator is None:
                raise NotFoundError(f"No account is registered under {email}")
            return collaborator
        collaborator = await self._users.get_by_username(session, username or "")
        if collaborator is None:
            raise NotFoundError("User not found")
        return collaborator

    async def _load_or_404(self, session: AsyncSession, project_id: UUID) -> Project:
        project = await self._projects.get_by_id(session, project_id)
        if project is None:
            raise NotFoundError("Project not found")
        return project

    def _require_member(self, project: Project, user_id: UUID) -> None:
        if not is_member(project, user_id):
            raise AccessDeniedError("You do not have access to this project")

    def _require_owner(self, project: Project, user_id: UUID) -> None:
        if not has_owner(project):
            raise AccessDeniedError("This project has no owner; owner-only actions are unavailable")
        if not is_owner(project, user_id):
            raise AccessDeniedError("Only the project owner can perform this action")
