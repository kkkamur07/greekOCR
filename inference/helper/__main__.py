"""Run the Inference helper sidecar on localhost."""

from __future__ import annotations

import logging
import socket
from logging.handlers import RotatingFileHandler
from pathlib import Path

from uvicorn import Config, Server

from inference.helper.app import create_helper_app
from inference.helper.settings import get_helper_settings


def bind_loopback_socket(port: int) -> socket.socket:
    """Listen on IPv4 loopback only. Clients reach the helper at 127.0.0.1."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(2048)
    return sock


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
    # HelperSettings rejects any non-loopback HELPER_HOST, so this always binds
    # to the local machine only.
    config = Config(
        create_helper_app(),
        host=settings.helper_host,
        port=settings.helper_port,
        log_level="info",
        log_config=None,
    )
    Server(config).run(sockets=[bind_loopback_socket(settings.helper_port)])


if __name__ == "__main__":
    main()
