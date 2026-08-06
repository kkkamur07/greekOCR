"""What the signed page image link is, and what it is deliberately not.

Asserted on the real ``create_app()``. ADR 0001 records what a test-local app
cost this project: a whole device layer that was never mounted, hidden behind a
green suite.

The rejected alternative is asserted here too. ADR 0002 turned down an
authenticated ``GET /device/v1/jobs/{id}/image`` for two reasons - a serverless
API should not stream manuscript scans, and the device credential should not
carry a route that must independently re-derive job ownership - and a rejected
alternative that nothing guards is one refactor away from being adopted by
accident.
"""

from __future__ import annotations

import os

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")

from datetime import UTC, datetime, timedelta

from fastapi.routing import APIRoute

from backend.core.app import create_app
from backend.document.infrastructure.media_store import (
    SIGNED_MEDIA_PREFIX,
    sign_object_path,
)
from backend.users.api.dependencies import get_current_user
from infrastructure.db import get_db

#: The route as Starlette holds it. OpenAPI drops the ``:path`` converter, so
#: the two spellings differ and both are asserted below.
SIGNED_MEDIA_PATH = f"{SIGNED_MEDIA_PREFIX}/{{image_key:path}}"
SIGNED_MEDIA_OPENAPI_PATH = f"{SIGNED_MEDIA_PREFIX}/{{image_key}}"


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


def test_the_signed_media_route_is_mounted_on_the_real_application() -> None:
    app = create_app()

    assert SIGNED_MEDIA_OPENAPI_PATH in set(app.openapi()["paths"])
    # A ``:path`` parameter, so one link can name a key with slashes in it -
    # ``parts/<uuid>/<stem>.webp`` is a legal key.
    assert _route(app, SIGNED_MEDIA_PATH, "GET") is not None


def test_the_signed_media_route_carries_no_credential_dependency() -> None:
    """The signature *is* the authorization. A session dependency here would mean
    the link needed something else too, which is a different design."""
    dependencies = _dependency_calls(_route(create_app(), SIGNED_MEDIA_PATH, "GET").dependant)

    assert get_current_user not in dependencies
    assert get_db not in dependencies


def test_the_walk_that_proves_it_can_actually_find_a_credential_dependency() -> None:
    """The control. Without it a broken walk passes the test above silently."""
    dependencies = _dependency_calls(
        _route(create_app(), "/media/parts/{part_id}", "GET").dependant
    )

    assert get_current_user in dependencies
    assert get_db in dependencies


def test_no_authenticated_page_image_route_was_added_to_the_device_credential() -> None:
    """The rejected alternative, asserted so it cannot be adopted quietly."""
    paths = set(create_app().openapi()["paths"])

    assert not [path for path in paths if path.startswith("/device/") and "image" in path]
    assert {path for path in paths if path.startswith("/device/v1/jobs")} == {
        "/device/v1/jobs/claim"
    }


def test_a_signature_binds_one_key_and_one_deadline() -> None:
    """The unit-level statement of what the integration suite proves over HTTP."""
    expires_at = datetime.now(UTC) + timedelta(seconds=60)
    key = "parts/11111111-1111-1111-1111-111111111111.webp"
    sibling = "parts/22222222-2222-2222-2222-222222222222.webp"

    signature = sign_object_path(key, expires_at=expires_at).split("signature=")[1]

    assert signature != sign_object_path(sibling, expires_at=expires_at).split("signature=")[1]
    assert (
        signature
        != sign_object_path(key, expires_at=expires_at + timedelta(seconds=1)).split("signature=")[
            1
        ]
    )
