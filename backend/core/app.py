"""FastAPI application factory — wires routers from core and bounded contexts."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.api.health import router as health_router
from backend.core.settings import get_app_settings


def create_app() -> FastAPI:
    app_settings = get_app_settings()
    app = FastAPI(
        title="greekOCR Platform",
        version="0.1.0",
        description="Greek manuscript OCR and annotation platform",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    # users, project, document, inference routers register here as issues land
    return app
