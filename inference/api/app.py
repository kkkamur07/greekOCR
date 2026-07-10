from fastapi import APIRouter, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from inference.admission import CLIENT_INPUT_ERROR
from inference.api.admission import RequestBodyLimitMiddleware, ServiceRateLimitMiddleware
from inference.api.health import router as health_router
from inference.api.jobs import router as jobs_router
from inference.api.run import router as run_router
from inference.infrastructure.settings import get_inference_settings

router = APIRouter(tags=["root"])


class RootResponse(BaseModel):
    message: str


@router.get("/", response_model=RootResponse, status_code=status.HTTP_200_OK)
def root() -> RootResponse:
    return RootResponse(message="Nomicous ML inference API")


def create_app() -> FastAPI:
    settings = get_inference_settings()
    settings.require_service_endpoint_configuration()
    app = FastAPI(title="nomicous ML inference service", version="0.1.0")
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=settings.inference_max_request_body_bytes,
    )
    app.add_middleware(
        ServiceRateLimitMiddleware,
        requests_per_minute=settings.inference_rate_limit_per_minute,
        service_secret=settings.inference_service_secret,
        limit_only_authenticated_service_requests=True,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": CLIENT_INPUT_ERROR},
        )

    app.include_router(router)
    app.include_router(health_router)
    app.include_router(jobs_router)
    app.include_router(run_router)
    return app
