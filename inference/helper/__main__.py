"""Run the Inference helper sidecar on localhost."""

from __future__ import annotations

import uvicorn

from inference.helper.app import create_helper_app
from inference.helper.settings import get_helper_settings


def main() -> None:
    settings = get_helper_settings()
    uvicorn.run(
        create_helper_app(),
        host=settings.helper_host,
        port=settings.helper_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
