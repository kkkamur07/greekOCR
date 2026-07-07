#!/usr/bin/env python3
"""Seed a dev user for local login (idempotent by email)."""

import asyncio
import os
import sys

# Annote app root on PYTHONPATH when run: PYTHONPATH=. python scripts/seed_dev_user.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.db import AsyncSessionLocal
from infrastructure import models as _orm_models  # noqa: F401 — register all mappers
from backend.dev.bootstrap import DEV_EMAIL, DEV_PASSWORD, reset_dev_user_password
from backend.users.application.auth_service import AuthService
PRINT_TOKEN = os.environ.get("SEED_DEV_PRINT_TOKEN", "").lower() in ("1", "true", "yes")


async def main() -> None:
    async with AsyncSessionLocal() as session:
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
