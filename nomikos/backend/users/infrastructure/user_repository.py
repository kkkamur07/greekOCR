"""User persistence."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.users.infrastructure.orm_models import User


class UserRepository:
    async def get_by_id(self, session: AsyncSession, user_id: UUID) -> User | None:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, session: AsyncSession, email: str) -> User | None:
        # Case-insensitive so the uniqueness check catches case variants and any
        # caller that did not pass through the request schema's normalisation
        # (device pairing, legacy mixed-case rows) still resolves to one account.
        result = await session.execute(
            select(User).where(func.lower(User.email) == func.lower(email))
        )
        return result.scalar_one_or_none()

    async def get_by_username(self, session: AsyncSession, username: str) -> User | None:
        result = await session.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def create(
        self,
        session: AsyncSession,
        *,
        email: str,
        username: str,
        hashed_password: str,
    ) -> User:
        """Stage a new user; caller commits the session."""
        user = User(email=email, username=username, hashed_password=hashed_password)
        session.add(user)
        await session.flush()
        await session.refresh(user)
        return user
