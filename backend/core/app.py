"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.health import router as health_router
from infrastructure.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="greekOCR Platform",
        version="0.1.0",
        description="Greek manuscript OCR and annotation platform",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    return app
