#!/usr/bin/env python3
"""Seed a dev user for local login (idempotent by email)."""

import asyncio
import os
import sys

# Annote app root on PYTHONPATH when run: PYTHONPATH=. python scripts/seed_dev_user.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.db import AsyncSessionLocal
from backend.users.application.auth_service import AuthService

DEV_EMAIL = os.environ.get("DEV_USER_EMAIL", "dev@kalamos.local")
DEV_USERNAME = os.environ.get("DEV_USER_USERNAME", "dev")
DEV_PASSWORD = os.environ.get("DEV_USER_PASSWORD", "dev-pass-123")
PRINT_TOKEN = os.environ.get("SEED_DEV_PRINT_TOKEN", "").lower() in ("1", "true", "yes")


async def main() -> None:
    async with AsyncSessionLocal() as session:
        service = AuthService()
        user, token = await service.register_if_absent(
            session,
            email=DEV_EMAIL,
            username=DEV_USERNAME,
            password=DEV_PASSWORD,
        )
        if user is None:
            print(f"Dev user already exists: {DEV_EMAIL}")
            return
        print(f"Created dev user id={user.id} email={DEV_EMAIL}")
        if PRINT_TOKEN and token:
            print(f"Sample JWT (first 40 chars): {token[:40]}...")


if __name__ == "__main__":
    asyncio.run(main())
