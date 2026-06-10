"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from annote import __version__
from annote.api.health import router as health_router
from annote.api.pages import router as pages_router
from annote.settings import get_settings
from annote.services.data_layout import ensure_data_directories


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    import logging

    logging.getLogger("annote").info("data_root=%s", settings.data_root)
    try:
        ensure_data_directories(settings.data_root)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create annote data directories under {settings.data_root}. "
            "Set ANNOTE_DATA_ROOT in backend/.env or create the path with write permission."
        ) from exc
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="annote",
        version=__version__,
        description="Standalone manuscript line annotation API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(pages_router)
    return app
