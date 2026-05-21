#!/usr/bin/env python3
"""Seed a dev user for local login (idempotent by email)."""

import asyncio
import os
import sys

# Repo root on PYTHONPATH when run: PYTHONPATH=. python scripts/seed_dev_user.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.db import AsyncSessionLocal
from backend.users.application.auth_service import AuthService

DEV_EMAIL = os.environ.get("DEV_USER_EMAIL", "dev@kalamos.local")
DEV_USERNAME = os.environ.get("DEV_USER_USERNAME", "dev")
DEV_PASSWORD = os.environ.get("DEV_USER_PASSWORD", "dev-pass-123")


async def main() -> None:
    async with AsyncSessionLocal() as session:
        service = AuthService()
        existing = await service._repo.get_by_email(session, DEV_EMAIL)
        if existing:
            print(f"Dev user already exists: {DEV_EMAIL}")
            return
        user, token = await service.register(
            session,
            email=DEV_EMAIL,
            username=DEV_USERNAME,
            password=DEV_PASSWORD,
        )
        print(f"Created dev user id={user.id} email={DEV_EMAIL}")
        print(f"Sample JWT (first 40 chars): {token[:40]}...")


if __name__ == "__main__":
    asyncio.run(main())
