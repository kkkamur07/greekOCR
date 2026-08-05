"""Helper process entrypoint binds IPv4 loopback only."""

from __future__ import annotations

import socket

import pytest
from inference.helper.__main__ import bind_loopback_socket


def _free_port() -> int:
    # Bind an ephemeral port so this stays hermetic and does not fight the
    # installed helper on :8001.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_bind_loopback_socket_listens_on_ipv4_loopback() -> None:
    port = _free_port()

    sock = bind_loopback_socket(port)
    try:
        assert sock.family == socket.AF_INET
        assert sock.getsockname()[0] == "127.0.0.1"
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    finally:
        sock.close()


def test_bind_loopback_socket_is_not_reachable_off_host() -> None:
    """A LAN address must not reach the helper socket."""
    port = _free_port()

    sock = bind_loopback_socket(port)
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
        if host_ip.startswith("127."):
            pytest.skip("host resolves to loopback; no routable address to probe")
        with pytest.raises(OSError):
            with socket.create_connection((host_ip, port), timeout=1):
                pass
    finally:
        sock.close()
