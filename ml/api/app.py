from fastapi import FastAPI

from ml.api.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(title="nomicous ML inference service", version="0.1.0")
    app.include_router(health_router)
    return app
