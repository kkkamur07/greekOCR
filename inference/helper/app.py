"""FastAPI app factory for the Inference helper sidecar."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.admission import CLIENT_INPUT_ERROR
from inference.api.admission import RequestBodyLimitMiddleware, ServiceRateLimitMiddleware
from inference.api.health import router as health_router
from inference.helper.routes.info import router as info_router
from inference.helper.routes.run import router as run_router
from inference.helper.settings import HELPER_VERSION, apply_helper_environment

HELPER_INTERNAL_ERROR = "Internal helper error"

logger = logging.getLogger(__name__)

ASGIApp = Callable[
    [dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]],
    Awaitable[None],
]


class UnhandledErrorMiddleware:
    """Convert escaping exceptions into JSON before ServerErrorMiddleware runs.

    Starlette's ``ServerErrorMiddleware`` sits outside user middleware (including
    CORS). An uncaught exception therefore becomes a bare 500 with no
    ``Access-Control-Allow-Origin``, which browsers report as a CORS failure.
    Catching here (inside CORS) keeps error bodies readable to the hosted app.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict,
        receive: Callable[[], Awaitable[dict]],
        send: Callable[[dict], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        except Exception:
            logger.exception("Unhandled inference helper error")
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": HELPER_INTERNAL_ERROR},
            )
            await response(scope, receive, send)


def create_helper_app() -> FastAPI:
    settings = apply_helper_environment()
    app = FastAPI(
        title="Nomicous Inference Helper",
        version=HELPER_VERSION,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=settings.inference_max_request_body_bytes,
    )
    app.add_middleware(
        ServiceRateLimitMiddleware,
        requests_per_minute=settings.inference_rate_limit_per_minute,
    )

    # Inside CORS so converted 500s still receive Access-Control-Allow-Origin.
    app.add_middleware(UnhandledErrorMiddleware)

    # Added last so CORS is the outermost middleware and 429/413/500 responses
    # still carry CORS headers that browser clients can read.
    # allow_private_network: Chrome/Edge send Access-Control-Request-Private-Network
    # on preflight for public HTTPS → loopback POSTs. Without this, GET /health and
    # /info succeed (simple requests, no preflight) while POST /inference/v1/run
    # fails in the browser as TypeError "Failed to fetch".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://app.nomicous.com"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_private_network=True,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": CLIENT_INPUT_ERROR},
        )

    app.include_router(health_router)
    app.include_router(info_router)
    app.include_router(run_router)
    return app
