#!/usr/bin/env python3
"""Seed a dev user for local login (idempotent by email)."""

import asyncio
import os

from _bootstrap import ensure_nomicous_on_path

ensure_nomicous_on_path()

from infrastructure.db import system_session
from infrastructure import models as _orm_models  # noqa: F401 — register all mappers
from backend.dev.bootstrap import DEV_EMAIL, DEV_PASSWORD, reset_dev_user_password
from backend.users.application.auth_service import AuthService

PRINT_TOKEN = os.environ.get("SEED_DEV_PRINT_TOKEN", "").lower() in ("1", "true", "yes")


async def main() -> None:
    async with system_session() as session:
        user_before = await AuthService().find_by_email(session, DEV_EMAIL)
        await reset_dev_user_password(session)
        if user_before is None:
            print(f"Created dev user email={DEV_EMAIL}")
        else:
            print(f"Reset dev user password: {DEV_EMAIL}")

        if PRINT_TOKEN:
            _user, token = await AuthService().login(
                session,
                email=DEV_EMAIL,
                password=DEV_PASSWORD,
            )
            print(f"Sample JWT (first 40 chars): {token[:40]}...")


if __name__ == "__main__":
    asyncio.run(main())
