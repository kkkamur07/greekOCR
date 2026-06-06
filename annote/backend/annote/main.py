"""Run annote API: `annote` after `pip install -e .`"""

import uvicorn

from annote.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "annote.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
