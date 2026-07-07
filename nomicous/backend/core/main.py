"""Uvicorn entrypoint: uvicorn backend.core.main:app"""

from backend.core.app import create_app

app = create_app()
