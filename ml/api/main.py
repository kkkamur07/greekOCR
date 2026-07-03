"""Uvicorn entrypoint: uvicorn ml.api.main:app."""

from ml.api.app import create_app

app = create_app()
