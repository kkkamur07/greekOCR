"""The claim endpoint, over HTTP, against the real app and live Postgres.

Everything here goes through ``create_app()``. ADR 0001 records why that is not
negotiable: an earlier device layer was never mounted, and the integration suite
hid it by building its own FastAPI app - a whole phase unreachable behind a green
suite. So there is no local app in this file, no substitute for Postgres, and the
device credential is minted by running the real pairing protocol rather than by
writing a row.

There are no assertions about how long a long poll takes. The contract is what a
claim returns and to whom; the clock is not part of it, and an assertion on it
would fail for reasons that are not defects.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from inference.contracts.webhooks import INFERENCE_WEBHOOK_SECRET_HEADER
from sqlalchemy import select

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.jobs.application.job_claim_service import agent_claim_owner
from backend.jobs.infrastructure.orm_models import Job, JobStatus
from backend.ml.application.agent_credentials import (
    SERVICE_ACCOUNT_ID,
    SERVICE_TOKEN_HEADER,
    WORKER_NAME_HEADER,
)
from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER
from backend.ml.domain.execution import ExecutionTarget
from backend.ml.infrastructure.device_orm_models import HelperDevice
from infrastructure.db import engine, sync_system_session
from tests.nomicous.integration.helpers import (
    MINIMAL_PNG,
    documents_url,
    pair_device_over_http,
    user_id_for_email,
)

pytestmark = pytest.mark.integration

CLAIM_URL = "/device/v1/jobs/claim"
CALLBACK_URL = "/internal/inference/job-complete"
SERVICE_TOKEN = "test-inference-worker-service-token-not-for-production"
SERVICE_HEADERS = {SERVICE_TOKEN_HEADER: SERVICE_TOKEN}


@pytest.fixture(scope="module", autouse=True)
def return_pooled_connections_before_leaving(client: TestClient):
    """Empty the async pool on the way out, on the loop that filled it.

    The concurrent claims below are the first thing in this suite to open more
    than a connection or two at once, so this module leaves an async pool full of
    live ``asyncpg`` connections bound to the session client's event loop.
    ``test_device_pairing.py`` then starts a *second* app on its own loop and
    inherits them, which is the collision tracked as issue #63 - a pre-existing
    bug, but one this file would otherwise widen from four tests to eleven.

    Disposing through ``client.portal`` runs the teardown on the loop that owns
    those connections, so they are closed rather than orphaned. It does not fix
    #63; it stops this module feeding it.
    """
    yield
    client.portal.call(engine.dispose)


# ---------------------------------------------------------------------------
# Live fixtures: real pairing, real capacity, real jobs
# ---------------------------------------------------------------------------


def _device_headers(token: str) -> dict[str, str]:
    return {DEVICE_TOKEN_HEADER: token}


def _claim(client: TestClient, headers: dict[str, str], *, wait_seconds: int = 0):
    return client.post(CLAIM_URL, headers=headers, json={"wait_seconds": wait_seconds})


def _running_agent(client: TestClient, headers: dict[str, str], name: str = "Laptop") -> dict:
    """Pair a laptop and let it announce itself the way the agent does: by asking
    for work. That first empty claim is what gives ``local`` **capacity**."""
    paired = pair_device_over_http(client, headers, name=name)
    empty = _claim(client, _device_headers(paired["device_token"]))
    assert empty.status_code == 200, empty.text
    assert empty.json()["page"] is None
    return paired


def _running_cloud_worker(client: TestClient, *, worker_name: str = "cloud-worker") -> dict:
    """A hosted worker registers itself by working. Its first claim provisions the
    ``cloud`` device row that reports cloud **capacity**."""
    response = client.post(
        CLAIM_URL,
        headers={**SERVICE_HEADERS, WORKER_NAME_HEADER: worker_name},
        json={"wait_seconds": 0},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _make_part(client: TestClient, headers: dict[str, str], project: dict) -> tuple[str, str]:
    base = documents_url(project["id"])
    created = client.post(base, headers=headers, json={"name": "Claimable page"})
    assert created.status_code == 201, created.text
    document_id = created.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201, upload.text
    return document_id, upload.json()["id"]


def _submit_segment(
    client: TestClient, headers: dict[str, str], project: dict, ids: tuple[str, str]
) -> str:
    document_id, part_id = ids
    response = client.post(
        f"{documents_url(project['id'])}/{document_id}/parts/{part_id}/segment",
        headers=headers,
    )
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def _prefer_local(client: TestClient, headers: dict[str, str]) -> None:
    response = client.put(
        "/account/execution-target", headers=headers, json={"prefer_local_inference": True}
    )
    assert response.status_code == 200, response.text


def _stored_job(job_id: str) -> Job:
    with sync_system_session() as session:
        return session.execute(select(Job).where(Job.id == uuid.UUID(job_id))).scalar_one()


# ---------------------------------------------------------------------------
# One claim returns at most one page
# ---------------------------------------------------------------------------


def test_one_claim_hands_over_exactly_one_page(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """A batch is N claims, not one claim of N pages."""
    agent = _running_agent(client, owner_headers)
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    first = _submit_segment(client, owner_headers, owner_project, ids)
    second = _submit_segment(client, owner_headers, owner_project, ids)

    response = _claim(client, _device_headers(agent["device_token"]))

    assert response.status_code == 200, response.text
    body = response.json()
    page = body["page"]
    assert page is not None
    assert page["product_job_id"] == first, "the oldest pending page goes first"
    assert page["job_type"] == "segment"
    assert page["execution_target"] == "local"
    # The claim carries the contract the inference runtime already takes, so the
    # agent runs the same code locally and in the cloud.
    assert page["request"]["product_job_id"] == first
    assert page["request"]["task"] == "segment"
    assert page["request"]["image_bytes"]
    assert page["inference_job_id"] == str(_stored_job(first).inference_job_id)
    assert body["poll_after_seconds"] == 0
    assert body["lease_seconds"] > 0

    # The second page is untouched and still claimable.
    assert _stored_job(second).status is JobStatus.pending
    assert _stored_job(first).status is JobStatus.waiting
    assert _stored_job(first).claimed_by == agent_claim_owner(uuid.UUID(agent["device_id"]))


def test_a_claimed_page_is_not_handed_out_twice(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    agent = _running_agent(client, owner_headers)
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)

    headers = _device_headers(agent["device_token"])
    assert _claim(client, headers).json()["page"]["product_job_id"] == job_id
    assert _claim(client, headers).json()["page"] is None


# ---------------------------------------------------------------------------
# Authorization: account scope, and execution target
# ---------------------------------------------------------------------------


def test_a_device_cannot_claim_another_accounts_work(
    client: TestClient,
    owner_user,
    owner_headers,
    outsider_user,
    outsider_headers,
    owner_project,
) -> None:
    """The device credential's scope is one ``helper_devices.user_id`` foreign key,
    and the queue predicate is that same column. Not a code-review promise."""
    owner_agent = _running_agent(client, owner_headers, name="Owner laptop")
    outsider_agent = _running_agent(client, outsider_headers, name="Outsider laptop")
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)

    stolen = _claim(client, _device_headers(outsider_agent["device_token"]))

    assert stolen.status_code == 200, stolen.text
    assert stolen.json()["page"] is None
    assert _stored_job(job_id).status is JobStatus.pending
    # And the rightful owner still gets it.
    assert (
        _claim(client, _device_headers(owner_agent["device_token"])).json()["page"][
            "product_job_id"
        ]
        == job_id
    )


def test_a_device_cannot_claim_cloud_work(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """``cloud`` work is platform work: it is not scoped by the one account a
    device token names, so honouring a device token for it would hand one
    laptop every account's pages."""
    agent = _running_agent(client, owner_headers)
    _running_cloud_worker(client)
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)
    assert _stored_job(job_id).execution_target is ExecutionTarget.cloud

    response = _claim(client, _device_headers(agent["device_token"]))

    assert response.status_code == 200, response.text
    assert response.json()["page"] is None
    assert _stored_job(job_id).status is JobStatus.pending


def test_a_service_credential_claims_cloud_work_from_the_same_endpoint(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """One agent implementation, two credentials - not two code paths kept in
    parity by discipline (ADR 0003)."""
    _running_cloud_worker(client)
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)

    response = client.post(CLAIM_URL, headers=SERVICE_HEADERS, json={"wait_seconds": 0})

    assert response.status_code == 200, response.text
    page = response.json()["page"]
    assert page is not None
    assert page["product_job_id"] == job_id
    assert page["execution_target"] == "cloud"
    assert _stored_job(job_id).status is JobStatus.waiting


def test_a_service_credential_cannot_claim_local_work(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The asymmetry runs both ways: a hosted worker never takes a page a
    researcher's own computer was chosen for."""
    _running_agent(client, owner_headers)
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)
    assert _stored_job(job_id).execution_target is ExecutionTarget.local

    response = client.post(CLAIM_URL, headers=SERVICE_HEADERS, json={"wait_seconds": 0})

    assert response.status_code == 200, response.text
    assert response.json()["page"] is None
    assert _stored_job(job_id).status is JobStatus.pending


def test_an_unauthenticated_or_wrong_credential_never_reaches_the_queue(
    client: TestClient, owner_headers
) -> None:
    assert client.post(CLAIM_URL, json={"wait_seconds": 0}).status_code == 401
    assert (
        client.post(
            CLAIM_URL, headers={DEVICE_TOKEN_HEADER: "nmd1.not-a-token"}, json={"wait_seconds": 0}
        ).status_code
        == 401
    )
    assert (
        client.post(
            CLAIM_URL,
            headers={SERVICE_TOKEN_HEADER: "wrong-service-token-but-long-enough-to-pass"},
            json={"wait_seconds": 0},
        ).status_code
        == 401
    )
    # A browser access token is not an agent credential either.
    bearer = owner_headers["Authorization"].split(" ", 1)[1]
    assert (
        client.post(
            CLAIM_URL, headers={DEVICE_TOKEN_HEADER: bearer}, json={"wait_seconds": 0}
        ).status_code
        == 401
    )


def test_a_revoked_device_stops_claiming_on_its_very_next_call(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    agent = _running_agent(client, owner_headers)
    revoked = client.delete(f"/devices/{agent['device_id']}", headers=owner_headers)
    assert revoked.status_code == 204, revoked.text

    assert _claim(client, _device_headers(agent["device_token"])).status_code == 401


# ---------------------------------------------------------------------------
# The service account, decided in this issue
# ---------------------------------------------------------------------------


def test_the_hosted_worker_is_owned_by_a_service_account_no_person_can_hold(
    client: TestClient, owner_user, owner_headers
) -> None:
    """``helper_devices.user_id`` is NOT NULL by design, so a hosted worker's row
    needs an owner - and that owner must not be a researcher, or an account
    deletion would take cloud inference with it."""
    _running_cloud_worker(client)

    with sync_system_session() as session:
        devices = (
            session.execute(
                select(HelperDevice).where(HelperDevice.inference_host == ExecutionTarget.cloud)
            )
            .scalars()
            .all()
        )
        assert len(devices) == 1
        worker = devices[0]
        assert worker.user_id == SERVICE_ACCOUNT_ID
        # It has no device credential at all: the service token is the only way
        # in, so revoking it is one environment variable rather than a row edit.
        assert worker.token_hash == ""
        assert worker.last_seen_at is not None

    # It never appears in a researcher's device list.
    assert client.get("/devices", headers=owner_headers).json() == []


def test_repeated_hosted_worker_claims_reuse_one_device_row(client: TestClient) -> None:
    for _ in range(3):
        _running_cloud_worker(client)
    _running_cloud_worker(client, worker_name="second-worker")

    with sync_system_session() as session:
        names = sorted(
            session.execute(
                select(HelperDevice.name).where(
                    HelperDevice.inference_host == ExecutionTarget.cloud
                )
            )
            .scalars()
            .all()
        )
    assert names == ["cloud-worker", "second-worker"]


# ---------------------------------------------------------------------------
# The empty queue
# ---------------------------------------------------------------------------


def test_an_empty_queue_is_a_well_formed_response_and_not_an_error(
    client: TestClient, owner_headers
) -> None:
    """An empty queue is the normal state of a healthy platform. A 404 here would
    teach every agent to treat "nothing to do" as a failure."""
    agent = _running_agent(client, owner_headers)

    response = _claim(client, _device_headers(agent["device_token"]))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["page"] is None
    assert body["poll_after_seconds"] > 0
    assert body["lease_seconds"] > 0
    assert body["server_time"]
    assert "error" not in body


def test_a_long_poll_over_an_empty_queue_still_answers_empty(
    client: TestClient, owner_headers
) -> None:
    """The contract, not the clock: what a long poll returns when nothing arrives."""
    agent = _running_agent(client, owner_headers)

    response = _claim(client, _device_headers(agent["device_token"]), wait_seconds=1)

    assert response.status_code == 200, response.text
    assert response.json()["page"] is None


def test_the_wait_is_clamped_rather_than_refused(client: TestClient, owner_headers) -> None:
    """An agent asking for an hour is answered, not rejected: the ceiling is the
    platform's dial, and the agent reads its cadence from the platform anyway."""
    agent = _running_agent(client, owner_headers)
    from backend.core.settings.device import get_device_settings

    assert get_device_settings().device_claim_max_wait_seconds <= 120
    response = client.post(
        CLAIM_URL,
        headers=_device_headers(agent["device_token"]),
        json={"wait_seconds": 3600},
    )
    # It returns; the clamp is what stops it taking an hour. No timing assertion:
    # the point is that an over-long request is served, not refused.
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Two agents never receive the same page
# ---------------------------------------------------------------------------


def test_two_agents_polling_concurrently_never_receive_the_same_page(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """``FOR UPDATE SKIP LOCKED`` against live Postgres, from real concurrent
    HTTP requests. A second claimer skips the locked row and takes the next one;
    it never waits on the first and never sees the same id."""
    first_agent = _running_agent(client, owner_headers, name="Laptop one")
    second_agent = _running_agent(client, owner_headers, name="Laptop two")
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    submitted = [_submit_segment(client, owner_headers, owner_project, ids) for _ in range(6)]

    tokens = [first_agent["device_token"], second_agent["device_token"]] * 4
    with ThreadPoolExecutor(max_workers=len(tokens)) as pool:
        responses = list(pool.map(lambda token: _claim(client, _device_headers(token)), tokens))

    assert all(response.status_code == 200 for response in responses)
    claimed = [
        response.json()["page"]["product_job_id"]
        for response in responses
        if response.json()["page"] is not None
    ]
    assert len(claimed) == len(set(claimed)), "the same page was handed to two agents"
    assert set(claimed) == set(submitted)
    with sync_system_session() as session:
        statuses = (
            session.execute(select(Job.status).where(Job.id.in_([uuid.UUID(j) for j in submitted])))
            .scalars()
            .all()
        )
    assert set(statuses) == {JobStatus.waiting}


def test_many_agents_can_long_poll_while_ordinary_traffic_is_served(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """More concurrent long polls than the connection pool has connections.

    ``DB_POOL_SIZE + DB_MAX_OVERFLOW`` is 15 on the defaults, which is the
    fifteen-device ceiling ADR 0001 names. A claim that took ``Depends(get_db)``
    would hold one of those for the whole poll; twenty of them would leave
    nothing for the researcher's own browser. The structural half of this
    guarantee - that the route declares no session dependency at all - is
    asserted in the unit suite, which is where a precise answer belongs.
    """
    agent = _running_agent(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    _prefer_local(client, owner_headers)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)
    # Claim it so the pollers below find an empty queue and actually wait.
    assert _claim(client, _device_headers(agent["device_token"])).json()["page"] is not None

    headers = _device_headers(agent["device_token"])
    with ThreadPoolExecutor(max_workers=21) as pool:
        pollers = [pool.submit(_claim, client, headers, wait_seconds=2) for _ in range(20)]
        browsing = pool.submit(client.get, f"/jobs/{job_id}", headers=owner_headers)
        read = browsing.result()
        claims = [poller.result() for poller in pollers]

    assert read.status_code == 200, read.text
    assert read.json()["status"] == "waiting"
    assert all(response.status_code == 200 for response in claims)
    assert all(response.json()["page"] is None for response in claims)


# ---------------------------------------------------------------------------
# Completion and failure go through the existing callback contract
# ---------------------------------------------------------------------------


def _segment_done_body(page: dict) -> dict:
    return {
        "inference_job_id": page["inference_job_id"],
        "product_job_id": page["product_job_id"],
        "task": "segment",
        "status": "done",
        "output": {
            "kind": "segment",
            "data": {
                "lines": [
                    {
                        "external_id": "l1",
                        "order": 0,
                        "baseline": {"type": "LineString", "coordinates": [[1, 1], [2, 1]]},
                        "points": [[1, 1], [2, 1], [2, 2], [1, 2]],
                    }
                ]
            },
        },
    }


def _segment_failed_body(page: dict) -> dict:
    return {
        "inference_job_id": page["inference_job_id"],
        "product_job_id": page["product_job_id"],
        "task": "segment",
        "status": "failed",
        "error": "the model would not load on this machine",
    }


def _claimed_page(client: TestClient, owner_headers, owner_project) -> tuple[dict, dict]:
    agent = _running_agent(client, owner_headers)
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    _submit_segment(client, owner_headers, owner_project, ids)
    page = _claim(client, _device_headers(agent["device_token"])).json()["page"]
    assert page is not None
    return agent, page


def test_completion_flows_through_the_existing_callback_contract(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """Not a new endpoint and not a new payload: the agent posts the same
    ``JobCallbackRequest`` the inference service always posted, and the same
    ``JobCallbackService`` merges it."""
    agent, page = _claimed_page(client, owner_headers, owner_project)

    response = client.post(
        CALLBACK_URL,
        headers=_device_headers(agent["device_token"]),
        json=_segment_done_body(page),
    )

    assert response.status_code == 204, response.text
    stored = _stored_job(page["product_job_id"])
    assert stored.status is JobStatus.done
    assert stored.result["lines_count"] == 1
    body = client.get(f"/jobs/{page['product_job_id']}", headers=owner_headers).json()
    assert body["status"] == "done"


def test_failure_flows_through_the_existing_callback_contract(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    agent, page = _claimed_page(client, owner_headers, owner_project)

    response = client.post(
        CALLBACK_URL,
        headers=_device_headers(agent["device_token"]),
        json=_segment_failed_body(page),
    )

    assert response.status_code == 204, response.text
    stored = _stored_job(page["product_job_id"])
    assert stored.status is JobStatus.failed
    assert "would not load" in stored.error


def test_an_agent_cannot_report_on_a_page_it_is_not_holding(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    agent, page = _claimed_page(client, owner_headers, owner_project)
    other = pair_device_over_http(client, owner_headers, name="Another laptop")

    response = client.post(
        CALLBACK_URL,
        headers=_device_headers(other["device_token"]),
        json=_segment_done_body(page),
    )

    assert response.status_code == 403, response.text
    assert _stored_job(page["product_job_id"]).status is JobStatus.waiting


def test_a_hosted_worker_reports_its_own_cloud_page(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    _running_cloud_worker(client)
    ids = _make_part(client, owner_headers, owner_project)
    _submit_segment(client, owner_headers, owner_project, ids)
    page = client.post(CLAIM_URL, headers=SERVICE_HEADERS, json={"wait_seconds": 0}).json()["page"]
    assert page is not None

    response = client.post(CALLBACK_URL, headers=SERVICE_HEADERS, json=_segment_done_body(page))

    assert response.status_code == 204, response.text
    assert _stored_job(page["product_job_id"]).status is JobStatus.done


def test_the_platform_webhook_credential_still_works_unchanged(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The internal callback path keeps its own credential and its own outcomes;
    the agent credentials are additions to it, not a replacement."""
    _agent, page = _claimed_page(client, owner_headers, owner_project)

    response = client.post(
        CALLBACK_URL,
        headers={INFERENCE_WEBHOOK_SECRET_HEADER: "test-inference-webhook-secret"},
        json=_segment_done_body(page),
    )

    assert response.status_code == 204, response.text
    assert _stored_job(page["product_job_id"]).status is JobStatus.done
