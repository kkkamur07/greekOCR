"""**Execution target** and **capacity** gating, over HTTP against live Postgres.

Everything here goes through the real ``create_app()`` and the real database.
ADR 0001 records why that is not negotiable: an earlier device layer was never
mounted, and the integration suite hid it by building its own FastAPI app in the
test file - an entire phase unreachable behind a green suite. So there is no
local app here, and no substitute for Postgres.

**Capacity is time-dependent, and it is controlled here by writing
``last_seen_at``**, not by patching a clock. That is the actual production
signal: a host has capacity when one of its devices was seen inside
``DEVICE_IDLE_WINDOW_SECONDS``. A test that froze time would prove the freezing
worked; a test that writes a timestamp proves the query does.

The four submission outcomes, which are the whole of this issue:

1. the preferred host has capacity - the job goes there and says so;
2. the preferred host does not and the other does - the job goes to the other
   *and says so*, never silently;
3. neither has capacity - submission is refused with a reason, rather than
   creating a job nobody will claim;
4. a submitted job's target cannot be changed - not through the mapper, and not
   through raw SQL either.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

import infrastructure.models  # noqa: F401 — register all ORM mappers
from backend.core.exceptions import ConflictError
from backend.core.settings.device import get_device_settings
from backend.jobs.infrastructure.job_repository import mark_job_failed
from backend.jobs.infrastructure.orm_models import Job, JobStatus
from backend.ml.domain.execution import ExecutionTarget
from infrastructure.db import sync_system_session
from tests.nomicous.integration.helpers import (
    MINIMAL_PNG,
    assert_api_error,
    documents_url,
    pair_inference_device,
    user_id_for_email,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Live fixtures: real rows, real timestamps
# ---------------------------------------------------------------------------


def _stale_seconds() -> float:
    """One second past the window that still counts a device as capacity."""
    return get_device_settings().device_idle_window_seconds + 1


def _make_part(client: TestClient, headers: dict[str, str], project: dict) -> tuple[str, str]:
    base = documents_url(project["id"])
    created = client.post(base, headers=headers, json={"name": "Execution target"})
    assert created.status_code == 201, created.text
    document_id = created.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201, upload.text
    return document_id, upload.json()["id"]


def _segment(client: TestClient, headers: dict[str, str], project: dict, ids: tuple[str, str]):
    document_id, part_id = ids
    return client.post(
        f"{documents_url(project['id'])}/{document_id}/parts/{part_id}/segment",
        headers=headers,
    )


def _stored_job(job_id: str) -> Job:
    with sync_system_session() as session:
        return session.execute(select(Job).where(Job.id == uuid.UUID(job_id))).scalar_one()


# ---------------------------------------------------------------------------
# Outcome 1: the preferred host has capacity
# ---------------------------------------------------------------------------


def test_a_job_runs_on_the_preferred_host_when_it_has_capacity(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]), host="cloud", seen_seconds_ago=5
    )
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    read = client.get(f"/jobs/{job_id}", headers=owner_headers)
    assert read.status_code == 200
    body = read.json()
    # The researcher is always told which host will run the job - including when
    # nothing unusual happened.
    assert body["execution_target"] == "cloud"
    assert body["preferred_execution_target"] == "cloud"
    assert body["execution_target_substituted"] is False
    assert _stored_job(job_id).execution_target is ExecutionTarget.cloud


def test_the_account_setting_sends_a_job_to_the_researchers_own_computer(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """ "Use my computer when it is available", and it is."""
    user_id = user_id_for_email(owner_user["email"])
    pair_inference_device(user_id=user_id, host="local", seen_seconds_ago=5)
    pair_inference_device(user_id=user_id, host="cloud", seen_seconds_ago=5)
    setting = client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )
    assert setting.status_code == 200, setting.text
    assert setting.json() == {
        "prefer_local_inference": True,
        "preferred_execution_target": "local",
        "available_targets": ["cloud", "local"],
    }
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 202, response.text
    body = client.get(f"/jobs/{response.json()['job_id']}", headers=owner_headers).json()
    assert body["execution_target"] == "local"
    assert body["execution_target_substituted"] is False


def test_another_researchers_laptop_is_not_capacity_for_this_account(
    client: TestClient,
    owner_user: dict[str, str],
    owner_headers: dict[str, str],
    outsider_user: dict[str, str],
    owner_project,
) -> None:
    """``local`` capacity is scoped to the owner; ``cloud`` is not. A stranger's
    running laptop must never make one of my jobs submittable."""
    pair_inference_device(
        user_id=user_id_for_email(outsider_user["email"]), host="local", seen_seconds_ago=5
    )
    client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 409, response.text
    assert (
        client.get("/account/execution-target", headers=owner_headers).json()["available_targets"]
        == []
    )


# ---------------------------------------------------------------------------
# Outcome 2: substituted, and the job says so
# ---------------------------------------------------------------------------


def test_an_unavailable_preferred_host_substitutes_and_the_job_reports_it(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """The laptop is paired but has not checked in inside the window, so it has no
    **capacity**. The job goes to the cloud - and the substitution is recorded on
    the job itself, not announced once into a toast the researcher may not see."""
    user_id = user_id_for_email(owner_user["email"])
    pair_inference_device(user_id=user_id, host="local", seen_seconds_ago=_stale_seconds())
    pair_inference_device(user_id=user_id, host="cloud", seen_seconds_ago=5)
    client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 202, response.text
    body = client.get(f"/jobs/{response.json()['job_id']}", headers=owner_headers).json()
    assert body["preferred_execution_target"] == "local"
    assert body["execution_target"] == "cloud"
    assert body["execution_target_substituted"] is True


def test_a_cloud_preference_substitutes_to_the_laptop_when_no_worker_is_online(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """Cloud capacity will not exist for some time, and that needs no special
    handling: a running laptop simply answers the same query."""
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]), host="local", seen_seconds_ago=5
    )
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 202, response.text
    body = client.get(f"/jobs/{response.json()['job_id']}", headers=owner_headers).json()
    assert body["preferred_execution_target"] == "cloud"
    assert body["execution_target"] == "local"
    assert body["execution_target_substituted"] is True


# ---------------------------------------------------------------------------
# Outcome 3: neither host has capacity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seen_seconds_ago", [None, 901.0])
def test_no_capacity_anywhere_refuses_the_submission_and_writes_no_job(
    client: TestClient,
    owner_user: dict[str, str],
    owner_headers: dict[str, str],
    owner_project,
    seen_seconds_ago: float | None,
) -> None:
    """Refusing is the point. A job created for a host nobody is claiming from
    has no terminal outcome - it is the failure mode ``local_only`` had, and it
    is why there is no hold window and no unclaimed-job sweeper."""
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]),
        host="local",
        seen_seconds_ago=seen_seconds_ago,
    )
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 409, response.text
    error = assert_api_error(response, code="CONFLICT")
    assert "No inference host is available" in error["message"]
    with sync_system_session() as session:
        assert session.execute(select(Job)).scalars().all() == []


def test_transcribe_is_gated_on_capacity_too(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    document_id, part_id = _make_part(client, owner_headers, owner_project)
    base = documents_url(owner_project["id"])
    lines = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "source": "manual",
                }
            ]
        },
    )
    assert lines.status_code == 200, lines.text

    refused = client.post(f"{base}/{document_id}/parts/{part_id}/transcribe", headers=owner_headers)
    assert refused.status_code == 409, refused.text

    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]), host="cloud", seen_seconds_ago=5
    )
    accepted = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe", headers=owner_headers
    )
    assert accepted.status_code == 202, accepted.text
    body = client.get(f"/jobs/{accepted.json()['job_id']}", headers=owner_headers).json()
    assert body["execution_target"] == "cloud"


# ---------------------------------------------------------------------------
# Outcome 4: a submitted job's target cannot be changed
# ---------------------------------------------------------------------------


def test_a_submitted_jobs_execution_target_cannot_be_changed_through_the_mapper(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]), host="cloud", seen_seconds_ago=5
    )
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _segment(client, owner_headers, owner_project, ids).json()["job_id"]

    with sync_system_session() as session:
        job = session.execute(select(Job).where(Job.id == uuid.UUID(job_id))).scalar_one()
        with pytest.raises(ConflictError, match="fixed when the job is submitted"):
            job.execution_target = ExecutionTarget.local
        session.rollback()

    assert _stored_job(job_id).execution_target is ExecutionTarget.cloud


def test_a_submitted_jobs_execution_target_cannot_be_changed_by_raw_sql(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """The application guard binds statements that go through the mapper. The
    platform also issues bulk ``UPDATE``s against ``jobs`` - the stale sweep, the
    callback path - so the rule is in Postgres as well."""
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]), host="cloud", seen_seconds_ago=5
    )
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _segment(client, owner_headers, owner_project, ids).json()["job_id"]

    with sync_system_session() as session:
        with pytest.raises(Exception, match="fixed at submission"):
            session.execute(
                text("UPDATE jobs SET execution_target = 'local' WHERE id = :id"),
                {"id": uuid.UUID(job_id)},
            )
        session.rollback()

    assert _stored_job(job_id).execution_target is ExecutionTarget.cloud


def test_an_ordinary_update_that_leaves_the_target_alone_still_works(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """The guard must be a guard, not a lock: every other column stays writable."""
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]), host="cloud", seen_seconds_ago=5
    )
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _segment(client, owner_headers, owner_project, ids).json()["job_id"]

    mark_job_failed(uuid.UUID(job_id), "inference failed", claimed_by=None)

    stored = _stored_job(job_id)
    assert stored.status is JobStatus.failed
    assert stored.execution_target is ExecutionTarget.cloud


# ---------------------------------------------------------------------------
# A failed job reports which host it failed on
# ---------------------------------------------------------------------------


def test_a_failed_job_reports_the_host_it_failed_on(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]), host="local", seen_seconds_ago=5
    )
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _segment(client, owner_headers, owner_project, ids).json()["job_id"]
    mark_job_failed(uuid.UUID(job_id), "inference failed", claimed_by=None)

    body = client.get(f"/jobs/{job_id}", headers=owner_headers).json()

    assert body["status"] == "failed"
    assert body["execution_target"] == "local"
    assert body["preferred_execution_target"] == "cloud"
    assert body["execution_target_substituted"] is True


# ---------------------------------------------------------------------------
# The account setting, and the absence of a per-job one
# ---------------------------------------------------------------------------


def test_the_execution_target_preference_persists_and_is_readable(
    client: TestClient, owner_headers: dict[str, str]
) -> None:
    assert client.get("/account/execution-target", headers=owner_headers).json() == {
        "prefer_local_inference": False,
        "preferred_execution_target": "cloud",
        "available_targets": [],
    }

    client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )

    assert client.get("/account/execution-target", headers=owner_headers).json() == {
        "prefer_local_inference": True,
        "preferred_execution_target": "local",
        "available_targets": [],
    }
    # Also on the account itself, so a client that already has the user does not
    # need a second round trip to render the setting.
    assert client.get("/me", headers=owner_headers).json()["prefer_local_inference"] is True


def test_the_preference_route_requires_a_logged_in_researcher(client: TestClient) -> None:
    assert client.get("/account/execution-target").status_code == 401
    assert (
        client.put("/account/execution-target", json={"prefer_local_inference": True}).status_code
        == 401
    )


def test_changing_the_preference_does_not_move_a_job_that_was_already_submitted(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    user_id = user_id_for_email(owner_user["email"])
    pair_inference_device(user_id=user_id, host="local", seen_seconds_ago=5)
    pair_inference_device(user_id=user_id, host="cloud", seen_seconds_ago=5)
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _segment(client, owner_headers, owner_project, ids).json()["job_id"]

    client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )

    body = client.get(f"/jobs/{job_id}", headers=owner_headers).json()
    assert body["execution_target"] == "cloud"


def test_there_is_no_per_job_execution_target_control(client: TestClient) -> None:
    """A per-job toggle is what regrows the three-mode complexity ADR 0002
    deletes, so its absence is asserted rather than assumed. ``local_only`` is
    absent from the published API for the same reason."""
    schema = client.get("/openapi.json").json()
    for name in ("SegmentPartRequest", "TranscribePartRequest"):
        properties = schema["components"]["schemas"][name].get("properties", {})
        assert not [key for key in properties if key.startswith("execution")]

    assert "local_only" not in client.get("/openapi.json").text
    assert set(schema["components"]["schemas"]["ExecutionTarget"]["enum"]) == {"local", "cloud"}
