"""Helper process entrypoint binds both loopback stacks."""

from __future__ import annotations

import socket

from inference.helper.__main__ import bind_loopback_sockets


def test_bind_loopback_sockets_listens_on_ipv4_and_ipv6() -> None:
    # Bind an ephemeral port so this stays hermetic and does not fight the
    # installed helper on :8001.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    sockets = bind_loopback_sockets(port)
    try:
        families = {sock.family for sock in sockets}
        assert socket.AF_INET in families
        assert socket.AF_INET6 in families or len(sockets) == 1

        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
        if socket.AF_INET6 in families:
            with socket.create_connection(("::1", port), timeout=1):
                pass
    finally:
        for sock in sockets:
            sock.close()
