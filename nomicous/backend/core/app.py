"""FastAPI application factory — wires routers from core and bounded contexts."""

import logging
from contextlib import asynccontextmanager, suppress
from http import HTTPStatus

import infrastructure.models  # noqa: F401 — register all ORM mappers before first query

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.annotation.api.history import router as annotation_history_router
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
from backend.core.schemas.errors import ApiErrorResponse
from backend.core.settings import get_app_settings
from backend.core.settings.job import get_job_settings
from backend.core.version import get_version
from backend.jobs.api.internal_inference import router as internal_inference_router
from backend.jobs.api.jobs import router as jobs_router
from backend.jobs.infrastructure.notifications import platform_job_notification_loop
from backend.jobs.infrastructure.worker import worker_loop
from backend.document.api.documents import router as documents_router
from backend.document.api.media import router as media_router
from backend.document.api.public import router as public_router
from backend.document.api.public_media import router as public_media_router
from backend.ml.api.models import router as ml_models_router
from backend.ml.api.registry import router as ml_registry_router
from backend.project.api.projects import router as projects_router
from backend.users.api.auth import router as auth_router

logger = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}

COMMON_ERROR_RESPONSES = {
    401: {"model": ApiErrorResponse, "description": "Not authenticated"},
    403: {"model": ApiErrorResponse, "description": "Not authorized"},
    404: {"model": ApiErrorResponse, "description": "Resource not found"},
    409: {"model": ApiErrorResponse, "description": "Conflict with current state"},
    422: {"model": ApiErrorResponse, "description": "Validation error"},
    429: {"model": ApiErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ApiErrorResponse, "description": "Internal server error"},
}


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: object | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    content: dict[str, object] = {"error": {"code": code, "message": message}}
    if details is not None:
        content["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=jsonable_encoder(content), headers=headers)


def _http_error_code(status_code: int) -> str:
    return {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
    }.get(status_code, "HTTP_ERROR")


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found_handler(_request: Request, exc: NotFoundError) -> JSONResponse:
        return _error_response(status_code=404, code="NOT_FOUND", message=str(exc))

    @app.exception_handler(InvalidCredentialsError)
    async def invalid_credentials_handler(
        _request: Request, exc: InvalidCredentialsError
    ) -> JSONResponse:
        return _error_response(status_code=401, code="UNAUTHORIZED", message=str(exc))

    @app.exception_handler(AccessDeniedError)
    async def access_denied_handler(_request: Request, exc: AccessDeniedError) -> JSONResponse:
        return _error_response(status_code=403, code="FORBIDDEN", message=str(exc))

    @app.exception_handler(ValidationError)
    async def validation_handler(_request: Request, exc: ValidationError) -> JSONResponse:
        return _error_response(status_code=422, code="VALIDATION_ERROR", message=str(exc))

    @app.exception_handler(ConflictError)
    async def conflict_handler(_request: Request, exc: ConflictError) -> JSONResponse:
        return _error_response(status_code=409, code="CONFLICT", message=str(exc))

    @app.exception_handler(GreekOCRException)
    async def greekocr_handler(_request: Request, exc: GreekOCRException) -> JSONResponse:
        logger.exception("Unhandled platform error", exc_info=exc)
        return _error_response(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        try:
            default_message = HTTPStatus(exc.status_code).phrase
        except ValueError:
            default_message = "Request failed"
        message = exc.detail if isinstance(exc.detail, str) else default_message
        details = None if isinstance(exc.detail, str) else exc.detail
        return _error_response(
            status_code=exc.status_code,
            code=_http_error_code(exc.status_code),
            message=message,
            details=details,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            status_code=422,
            code="VALIDATION_ERROR",
            message="Invalid request",
            details=exc.errors(),
        )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    import asyncio

    from infrastructure.db import engine, sync_engine

    stop_event = asyncio.Event()
    job_settings = get_job_settings()
    notification_task = (
        asyncio.create_task(platform_job_notification_loop(stop_event))
        if job_settings.job_sse_notifications_enabled
        else None
    )
    worker_task = (
        asyncio.create_task(worker_loop(stop_event))
        if job_settings.job_worker_enabled
        else None
    )
    yield
    stop_event.set()
    if worker_task is not None:
        worker_task.cancel()
    if notification_task is not None:
        notification_task.cancel()
    if notification_task is not None:
        with suppress(asyncio.CancelledError):
            await notification_task
    if worker_task is not None:
        with suppress(asyncio.CancelledError):
            await worker_task
    await engine.dispose()
    sync_engine.dispose()


def create_app() -> FastAPI:
    app_settings = get_app_settings()
    if app_settings.behind_proxy and not app_settings.forwarded_allow_ips:
        logger.warning(
            "BEHIND_PROXY is enabled but FORWARDED_ALLOW_IPS is not set; "
            "configure proxy header trust before production deployment."
        )
    app = FastAPI(
        title="greekOCR Platform",
        version=get_version(),
        description="Greek manuscript OCR and annotation platform",
        lifespan=_lifespan,
        responses=COMMON_ERROR_RESPONSES,
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    _register_exception_handlers(app)
    app.include_router(root_router)
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(projects_router)
    app.include_router(documents_router)
    app.include_router(annotation_history_router)
    app.include_router(media_router)
    app.include_router(public_router)
    app.include_router(public_media_router)
    app.include_router(ml_models_router)
    app.include_router(ml_registry_router)
    app.include_router(jobs_router)
    app.include_router(internal_inference_router)
    return app
