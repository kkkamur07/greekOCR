from fastapi import APIRouter, FastAPI, status
from pydantic import BaseModel

from inference.api.health import router as health_router
from inference.api.jobs import router as jobs_router
from inference.api.run import router as run_router

router = APIRouter(tags=["root"])


class RootResponse(BaseModel):
    message: str


@router.get("/", response_model=RootResponse, status_code=status.HTTP_200_OK)
def root() -> RootResponse:
    return RootResponse(message="Nomicous ML inference API")


def create_app() -> FastAPI:
    app = FastAPI(title="nomicous ML inference service", version="0.1.0")
    app.include_router(router)
    app.include_router(health_router)
    app.include_router(jobs_router)
    app.include_router(run_router)
    return app
