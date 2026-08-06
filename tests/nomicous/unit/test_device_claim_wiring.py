"""The claim endpoint's two structural guarantees, asserted on the real app.

1. **It is mounted.** ADR 0001 records what unmounted device routes cost: a whole
   phase was unreachable behind a green suite because the integration tests built
   their own FastAPI app. So this asks ``create_app()``.

2. **It declares no request-scoped database session.** ``get_db`` pins a pooled
   connection for the length of the request. A 25-second long poll per idle agent
   would exhaust the pool at ``DB_POOL_SIZE + DB_MAX_OVERFLOW`` - fifteen on the
   defaults - and ADR 0003 puts *all* inference on this path, so that ceiling
   binds sooner than it would for laptops alone.

   This is asserted structurally rather than by timing a pool. A structural
   assertion names the defect precisely ("someone added ``Depends(get_db)``")
   and cannot flake; a load test would only say "it got slow". The control
   assertion below - that the same walk *does* find ``get_db`` on a route that
   takes one - is what stops this quietly passing because the walk is broken.
"""

from __future__ import annotations

import os

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")

from fastapi.routing import APIRoute

from backend.core.app import create_app
from infrastructure.db import get_db

CLAIM_PATH = "/device/v1/jobs/claim"


def _dependency_calls(dependant) -> set:
    found = {dependant.call} if dependant.call is not None else set()
    for sub in dependant.dependencies:
        found |= _dependency_calls(sub)
    return found


def _route(app, path: str, method: str) -> APIRoute:
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path and method in route.methods:
            return route
    raise AssertionError(f"{method} {path} is not mounted on the real application")


def test_the_claim_route_is_mounted_on_the_real_application() -> None:
    app = create_app()
    assert CLAIM_PATH in set(app.openapi()["paths"])
    assert "post" in app.openapi()["paths"][CLAIM_PATH]


def test_the_claim_route_takes_no_request_scoped_database_session() -> None:
    app = create_app()

    assert get_db not in _dependency_calls(_route(app, CLAIM_PATH, "POST").dependant)


def test_the_walk_that_proves_it_can_actually_find_a_session_dependency() -> None:
    """The control. Without it, a broken walk would pass the test above silently."""
    app = create_app()

    assert get_db in _dependency_calls(_route(app, "/device/v1/self", "GET").dependant)
