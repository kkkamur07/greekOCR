"""Uvicorn entrypoint: uvicorn inference.api.main:app."""

from inference.api.app import create_app

app = create_app()
