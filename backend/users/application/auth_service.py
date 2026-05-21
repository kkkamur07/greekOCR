"""Register, login, and user lookup."""

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, NotFoundError
from backend.core.settings.auth import AuthSettings, get_auth_settings
from backend.users.application.jwt_tokens import create_access_token
from backend.users.application.password import hash_password, verify_password
from backend.users.infrastructure.orm_models import User
from backend.users.infrastructure.user_repository import UserRepository


class AuthService:
    def __init__(
        self,
        repository: UserRepository | None = None,
        auth_settings: AuthSettings | None = None,
    ) -> None:
        self._repo = repository or UserRepository()
        self._auth_settings = auth_settings or get_auth_settings()

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
        if user is None or not verify_password(password, user.hashed_password):
            raise NotFoundError("Invalid email or password")
        token = create_access_token(user.id, self._auth_settings)
        return user, token

    async def get_user(self, session: AsyncSession, user_id) -> User:
        user = await self._repo.get_by_id(session, user_id)
        if user is None:
            raise NotFoundError("User not found")
        return user
