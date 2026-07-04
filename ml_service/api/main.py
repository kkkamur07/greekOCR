"""Uvicorn entrypoint: uvicorn ml_service.api.main:app."""

from ml_service.api.app import create_app

app = create_app()
