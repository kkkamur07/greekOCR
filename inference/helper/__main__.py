"""Run the Inference helper sidecar on localhost."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn

from inference.helper.app import create_helper_app
from inference.helper.settings import get_helper_settings


def main() -> None:
    settings = get_helper_settings()
    log_path = Path.home() / ".nomicous" / "logs" / "inference-helper.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        handlers=[
            RotatingFileHandler(
                log_path,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
        ],
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    uvicorn.run(
        create_helper_app(),
        host=settings.helper_host,
        port=settings.helper_port,
        log_level="info",
        log_config=None,
    )


if __name__ == "__main__":
    main()
