"""Register, login, and user lookup."""

from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, InvalidCredentialsError, NotFoundError
from backend.core.settings.auth import AuthSettings, get_auth_settings
from backend.users.application.jwt_tokens import create_access_token
from backend.users.application.password import hash_password, verify_password
from backend.users.infrastructure.orm_models import User
from backend.users.infrastructure.user_repository import UserRepository

_DUMMY_PASSWORD_HASH = "$2b$12$t7YSQy5g4YoP4Bfr5DXh0eUg2kUE4qavr20ibunY9EEWibESvTARu"


class AuthService:
    def __init__(
        self,
        repository: UserRepository | None = None,
        auth_settings: AuthSettings | None = None,
    ) -> None:
        self._repo = repository or UserRepository()
        self._auth_settings = auth_settings or get_auth_settings()

    async def find_by_email(self, session: AsyncSession, email: str) -> User | None:
        return await self._repo.get_by_email(session, email)

    async def register(
        self,
        session: AsyncSession,
        *,
        email: str,
        username: str,
        password: str,
    ) -> tuple[User, str]:
        if await self._repo.get_by_email(session, email):
            raise ConflictError("Email already registered")
        if await self._repo.get_by_username(session, username):
            raise ConflictError("Username already taken")
        user = await self._repo.create(
            session,
            email=email,
            username=username,
            hashed_password=hash_password(password),
        )
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            # Pre-checks above return precise feedback; this catches concurrent inserts safely.
            raise ConflictError("Email or username already taken") from exc
        token = create_access_token(user.id, self._auth_settings)
        return user, token

    async def login(
        self,
        session: AsyncSession,
        *,
        email: str,
        password: str,
    ) -> tuple[User, str]:
        user = await self._repo.get_by_email(session, email)
        hash_to_check = user.hashed_password if user is not None else _DUMMY_PASSWORD_HASH
        password_matches = verify_password(password, hash_to_check)
        if user is None or not password_matches:
            raise InvalidCredentialsError("Invalid email or password")
        token = create_access_token(user.id, self._auth_settings)
        return user, token

    async def get_user(self, session: AsyncSession, user_id: UUID) -> User:
        user = await self._repo.get_by_id(session, user_id)
        if user is None:
            raise NotFoundError("User not found")
        return user

    async def register_if_absent(
        self,
        session: AsyncSession,
        *,
        email: str,
        username: str,
        password: str,
    ) -> tuple[User | None, str | None]:
        """Create user when email is free; return (None, None) if already exists."""
        if await self.find_by_email(session, email):
            return None, None
        user, token = await self.register(
            session,
            email=email,
            username=username,
            password=password,
        )
        return user, token
