"""Run the Inference helper sidecar on localhost."""

from __future__ import annotations

import logging
import socket
from logging.handlers import RotatingFileHandler
from pathlib import Path

from uvicorn import Config, Server

from inference.helper.app import create_helper_app
from inference.helper.settings import _is_loopback_host, get_helper_settings

logger = logging.getLogger(__name__)


def bind_loopback_sockets(port: int) -> list[socket.socket]:
    """Listen on both IPv4 and IPv6 loopback so 127.0.0.1 / localhost / [::1] work."""
    sockets: list[socket.socket] = []
    for family, address in (
        (socket.AF_INET, "127.0.0.1"),
        (socket.AF_INET6, "::1"),
    ):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((address, port))
            sock.listen(2048)
            sockets.append(sock)
        except OSError as exc:
            logger.warning("Could not bind helper on [%s]:%s (%s)", address, port, exc)
    if not sockets:
        raise RuntimeError(f"Could not bind helper on any loopback interface port {port}")
    return sockets


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
    config = Config(
        create_helper_app(),
        host=settings.helper_host,
        port=settings.helper_port,
        log_level="info",
        log_config=None,
    )
    server = Server(config)
    if _is_loopback_host(settings.helper_host):
        server.run(sockets=bind_loopback_sockets(settings.helper_port))
    else:
        server.run()


if __name__ == "__main__":
    main()
