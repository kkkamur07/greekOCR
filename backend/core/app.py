"""FastAPI application factory — wires routers from core and bounded contexts."""

import logging
from contextlib import asynccontextmanager

import infrastructure.models  # noqa: F401 — register all ORM mappers before first query

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.api.health import router as health_router
from backend.core.api.root import router as root_router
from backend.core.exceptions import (
    AccessDeniedError,
    ConflictError,
    GreekOCRException,
    InvalidCredentialsError,
    NotFoundError,
    ValidationError,
)
from backend.core.settings import get_app_settings
from backend.inference.api.jobs import router as jobs_router
from backend.inference.infrastructure.worker import worker_loop
from backend.document.api.documents import router as documents_router
from backend.document.api.media import router as media_router
from backend.project.api.projects import router as projects_router
from backend.users.api.auth import router as auth_router

logger = logging.getLogger(__name__)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found_handler(_request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidCredentialsError)
    async def invalid_credentials_handler(
        _request: Request, exc: InvalidCredentialsError
    ) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.exception_handler(AccessDeniedError)
    async def access_denied_handler(_request: Request, exc: AccessDeniedError) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(ValidationError)
    async def validation_handler(_request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def conflict_handler(_request: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(GreekOCRException)
    async def greekocr_handler(_request: Request, exc: GreekOCRException) -> JSONResponse:
        logger.exception("Unhandled platform error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@asynccontextmanager
async def _lifespan(app: FastAPI):
    import asyncio

    stop_event = asyncio.Event()
    worker_task = asyncio.create_task(worker_loop(stop_event))
    yield
    stop_event.set()
    await worker_task


def create_app() -> FastAPI:
    app_settings = get_app_settings()
    app = FastAPI(
        title="greekOCR Platform",
        version="0.1.0",
        description="Greek manuscript OCR and annotation platform",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _register_exception_handlers(app)
    app.include_router(root_router)
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(projects_router)
    app.include_router(documents_router)
    app.include_router(media_router)
    app.include_router(jobs_router)
    return app
