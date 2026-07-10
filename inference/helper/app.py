"""FastAPI app factory for the Inference helper sidecar."""

from __future__ import annotations

import secrets

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.admission import CLIENT_INPUT_ERROR
from inference.api.admission import RequestBodyLimitMiddleware, ServiceRateLimitMiddleware
from inference.api.health import router as health_router
from inference.helper.routes.cache import router as cache_router
from inference.helper.routes.catalog import router as catalog_router
from inference.helper.routes.run import router as run_router
from inference.helper.settings import apply_helper_environment

HELPER_AUTH_SECRET_HEADER = "X-Inference-Helper-Secret"


def create_helper_app() -> FastAPI:
    settings = apply_helper_environment()
    app = FastAPI(title="Nomicous Inference Helper", version="0.1.3")
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=settings.inference_max_request_body_bytes,
    )
    app.add_middleware(
        ServiceRateLimitMiddleware,
        requests_per_minute=settings.inference_rate_limit_per_minute,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.helper_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", HELPER_AUTH_SECRET_HEADER],
    )

    @app.middleware("http")
    async def require_helper_secret(request: Request, call_next):
        if settings.helper_secure_mode and request.method != "OPTIONS":
            supplied = request.headers.get(HELPER_AUTH_SECRET_HEADER)
            if supplied is None or not secrets.compare_digest(
                supplied, settings.helper_auth_secret or ""
            ):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Helper authentication required"},
                )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": CLIENT_INPUT_ERROR},
        )

    app.include_router(health_router)
    app.include_router(catalog_router)
    app.include_router(cache_router)
    app.include_router(run_router)
    return app
