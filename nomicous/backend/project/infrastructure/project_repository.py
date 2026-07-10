"""Project persistence."""

from uuid import UUID

from sqlalchemy import or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.core.api.pagination import PageCursor
from backend.project.infrastructure.orm_models import Project, project_shared_users


class ProjectRepository:
    async def get_by_id(self, session: AsyncSession, project_id: UUID) -> Project | None:
        result = await session.execute(
            select(Project)
            .options(selectinload(Project.shared_users))
            .where(Project.id == project_id)
        )
        return result.scalar_one_or_none()

    async def get_by_slug(self, session: AsyncSession, slug: str) -> Project | None:
        result = await session.execute(select(Project).where(Project.slug == slug))
        return result.scalar_one_or_none()

    async def list_for_user(
        self,
        session: AsyncSession,
        user_id: UUID,
        *,
        limit: int = 50,
        cursor: PageCursor | None = None,
    ) -> list[Project]:
        shared_ids = select(project_shared_users.c.project_id).where(
            project_shared_users.c.user_id == user_id
        )
        stmt = (
            select(Project)
            .where(or_(Project.owner_id == user_id, Project.id.in_(shared_ids)))
            .order_by(Project.created_at.desc(), Project.id.desc())
        )
        if cursor is not None:
            stmt = stmt.where(
                tuple_(Project.created_at, Project.id) < (cursor.created_at, cursor.id)
            )
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def create(
        self,
        session: AsyncSession,
        *,
        name: str,
        slug: str,
        guidelines: str | None,
        owner_id: UUID,
    ) -> Project:
        project = Project(name=name, slug=slug, guidelines=guidelines, owner_id=owner_id)
        session.add(project)
        await session.commit()
        await session.refresh(project)
        return project

    async def update(
        self, session: AsyncSession, project: Project, **fields: str | None
    ) -> Project:
        for key, value in fields.items():
            setattr(project, key, value)
        await session.commit()
        await session.refresh(project)
        return project

    async def delete(self, session: AsyncSession, project: Project) -> None:
        await session.delete(project)
        await session.commit()

    async def add_shared_user(self, session: AsyncSession, project: Project, user) -> None:
        if user not in project.shared_users:
            project.shared_users.append(user)
            await session.commit()

    async def remove_shared_user(self, session: AsyncSession, project: Project, user) -> bool:
        if user not in project.shared_users:
            return False
        project.shared_users.remove(user)
        await session.commit()
        return True
