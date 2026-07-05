"""Development-only bootstrap helpers (idempotent, no DB restarts)."""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession

from backend.users.application.auth_service import AuthService
from backend.users.application.password import hash_password

# Must pass Pydantic EmailStr (`.local` / `.test` TLDs are rejected).
DEV_EMAIL = os.environ.get("DEV_USER_EMAIL", "dev@example.com")
DEV_USERNAME = os.environ.get("DEV_USER_USERNAME", "dev")
DEV_PASSWORD = os.environ.get("DEV_USER_PASSWORD", "dev-pass-123")


async def ensure_dev_user_exists(session: AsyncSession) -> bool:
    """Create the dev user when missing. Returns True if a user was created."""
    service = AuthService()
    if await service.find_by_email(session, DEV_EMAIL):
        return False
    await service.register(
        session,
        email=DEV_EMAIL,
        username=DEV_USERNAME,
        password=DEV_PASSWORD,
    )
    return True


async def reset_dev_user_password(session: AsyncSession) -> None:
    """Ensure dev user exists and reset its password to the configured dev default."""
    service = AuthService()
    user = await service.find_by_email(session, DEV_EMAIL)
    if user is None:
        await service.register(
            session,
            email=DEV_EMAIL,
            username=DEV_USERNAME,
            password=DEV_PASSWORD,
        )
        return
    user.hashed_password = hash_password(DEV_PASSWORD)
    await session.commit()
