"""FastAPI app factory for the Inference helper sidecar."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference.api.health import router as health_router
from inference.helper.routes.cache import router as cache_router
from inference.helper.routes.catalog import router as catalog_router
from inference.helper.routes.run import router as run_router
from inference.helper.settings import apply_helper_environment


def create_helper_app() -> FastAPI:
    settings = apply_helper_environment()
    app = FastAPI(title="Nomicous Inference Helper", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.helper_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(catalog_router)
    app.include_router(cache_router)
    app.include_router(run_router)
    return app
