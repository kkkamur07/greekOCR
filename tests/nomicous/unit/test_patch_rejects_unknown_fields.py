"""A PATCH must refuse a field it cannot write, not accept the request and drop it.

The handlers apply ``model_dump(exclude_unset=True)`` verbatim with ``setattr``, and
``patch_fields.reject_unknown_fields`` was written to make an unrecognised key loud
rather than silent. It could not: pydantic's default is to drop extra keys, so by the
time the handler saw the dict the offending key was gone, and every one of that guard's
call sites was unreachable from HTTP. A client that misspelled a field got 200 and a
response body showing the value it thought it had just changed, unchanged.

Asserted through the mounted application, because that is where the drop happened - the
guard itself always behaved, which is why nothing noticed.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from backend.core.app import create_app
from backend.users.api.dependencies import get_current_user
from infrastructure.db import get_db

PROJECT_ID = uuid.uuid4()
DOCUMENT_ID = uuid.uuid4()
PART_ID = uuid.uuid4()
BLOCK_ID = uuid.uuid4()
LINE_ID = uuid.uuid4()

_BASE = f"/projects/{PROJECT_ID}/documents/{DOCUMENT_ID}"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """The real app with authentication and the session satisfied.

    Both are overridden rather than exercised: an unknown key has to be refused before
    anything reaches a repository, so a handler that ran at all would be the failure.
    """
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: None
    return TestClient(app)


@pytest.mark.parametrize(
    ("url", "body"),
    [
        (_BASE, {"nmae": "typo"}),
        (f"{_BASE}/parts/{PART_ID}/blocks/{BLOCK_ID}", {"boxes": {}}),
        (f"{_BASE}/parts/{PART_ID}/lines/{LINE_ID}", {"blockId": None}),
    ],
    ids=["document", "block", "line"],
)
def test_an_unrecognised_patch_field_is_refused(client, url, body) -> None:
    response = client.patch(url, json=body)

    assert response.status_code == 422
    # The platform answers validation failures with a fixed public message and a
    # correlation id; the offending key stays in the log. "Loud" here means the
    # request is rejected, not that the response echoes what the client sent.
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    ("url", "body"),
    [
        (_BASE, {"name": "Renamed", "workflow": "published", "extra": 1}),
        (f"{_BASE}/parts/{PART_ID}/blocks/{BLOCK_ID}", {"order": 0, "extra": 1}),
        (f"{_BASE}/parts/{PART_ID}/lines/{LINE_ID}", {"order": 0, "extra": 1}),
    ],
    ids=["document", "block", "line"],
)
def test_a_valid_field_does_not_excuse_an_unknown_one_beside_it(client, url, body) -> None:
    """The silent drop was worst here: the write succeeded, so nothing looked wrong."""
    assert client.patch(url, json=body).status_code == 422
