"""The device **lease**, end to end, against live Postgres and the real app.

Everything here goes through ``create_app()``. ADR 0001 records why that is not
negotiable: an earlier device layer was never mounted and the integration suite
hid it behind its own FastAPI app. So there is no local app in this file, no
substitute for Postgres, and the device credentials are minted by running the
real pairing protocol.

**Expiry is created by writing the timestamp, not by waiting for it.** The lease
is 600 seconds; a test that slept through one is a test nobody runs, and a
patched clock would prove something about the patch rather than about the SQL
that Postgres will actually execute in production. So the row's ``updated_at`` is
aged in the database and the real sweep is then asked what it makes of it.

The one thing this file exists to prove: an expired lease sends the page **back
to the queue**, not to ``failed``. ADR 0005 made a claimed page ``waiting`` so
``JobCallbackService`` needed no change, which quietly put a researcher's laptop
under the 240-second waiting timeout - a timeout that *fails* the job. A closed
lid is not a failed job.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.core.settings.device import get_device_settings
from backend.core.settings.job import get_job_settings
from backend.jobs.application.job_claim_service import agent_claim_owner
from backend.jobs.infrastructure.job_repository import (
    WAITING_TIMEOUT_ERROR,
    fail_stale_waiting_jobs,
    release_expired_device_leases,
)
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.jobs.infrastructure.stale_sweep import (
    reset_stale_sweep_throttle,
    run_stale_job_sweep,
)
from backend.ml.application.agent_credentials import SERVICE_TOKEN_HEADER, WORKER_NAME_HEADER
from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER
from infrastructure.db import engine, sync_system_session
from tests.nomicous.integration.helpers import (
    MINIMAL_PNG,
    documents_url,
    pair_device_over_http,
)

pytestmark = pytest.mark.integration

CLAIM_URL = "/device/v1/jobs/claim"
CALLBACK_URL = "/internal/inference/job-complete"
SERVICE_HEADERS = {SERVICE_TOKEN_HEADER: "test-inference-worker-service-token-not-for-production"}


@pytest.fixture(scope="module", autouse=True)
def return_pooled_connections_before_leaving(client: TestClient):
    """Empty the async pool on the way out, on the loop that filled it.

    Same reasoning as ``test_device_claim.py``: this module opens several
    connections at once, and leaving them bound to the session client's loop
    feeds the pre-existing asyncpg collision tracked as issue #63.
    """
    yield
    client.portal.call(engine.dispose)


@pytest.fixture(autouse=True)
def _clean_sweep_throttle():
    """The throttle is process-global; never let it leak between tests."""
    reset_stale_sweep_throttle()
    yield
    reset_stale_sweep_throttle()


# ---------------------------------------------------------------------------
# Live fixtures: real pairing, real capacity, real jobs
# ---------------------------------------------------------------------------


def _device_headers(token: str) -> dict[str, str]:
    return {DEVICE_TOKEN_HEADER: token}


def _claim(client: TestClient, token: str):
    return client.post(CLAIM_URL, headers=_device_headers(token), json={"wait_seconds": 0})


def _claim_allowed_to_sweep(client: TestClient, token: str):
    """A claim that is not sitting inside another claim's throttle window.

    ``sweep_stale_jobs_on_read`` runs at most once per process per
    ``JOB_STALE_SWEEP_MIN_INTERVAL_SECONDS`` - thirty seconds - so a hot endpoint
    cannot become a sweep loop. These tests make several claims inside one
    second, so the window is opened explicitly at the point where the recovery is
    the thing under test. That bounds detection lag in production by the
    throttle, not by the lease, and the throttle has its own tests; asserting on
    it here would be measuring the wrong mechanism.
    """
    reset_stale_sweep_throttle()
    return _claim(client, token)


def _running_agent(client: TestClient, headers: dict[str, str], name: str) -> dict:
    """Pair a laptop and let it announce itself the way the agent does: by asking
    for work. That first empty claim is what gives ``local`` **capacity**."""
    paired = pair_device_over_http(client, headers, name=name)
    empty = _claim(client, paired["device_token"])
    assert empty.status_code == 200, empty.text
    assert empty.json()["page"] is None
    return paired


def _prefer_local(client: TestClient, headers: dict[str, str]) -> None:
    response = client.put(
        "/account/execution-target", headers=headers, json={"prefer_local_inference": True}
    )
    assert response.status_code == 200, response.text


def _make_part(client: TestClient, headers: dict[str, str], project: dict) -> tuple[str, str]:
    base = documents_url(project["id"])
    created = client.post(base, headers=headers, json={"name": "Leased page"})
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


def _stored_job(job_id: str | uuid.UUID) -> Job:
    key = job_id if isinstance(job_id, uuid.UUID) else uuid.UUID(job_id)
    with sync_system_session() as session:
        return session.execute(select(Job).where(Job.id == key)).scalar_one()


def _age_by(job_id: str | uuid.UUID, seconds: float) -> None:
    """Move the row's clock back, in the database, by writing ``updated_at``.

    This is what a laptop that went to sleep ``seconds`` ago looks like to
    Postgres. ``updated_at`` is the column both waiting sweeps compare against,
    and an explicit value overrides the ORM's ``onupdate``.
    """
    key = job_id if isinstance(job_id, uuid.UUID) else uuid.UUID(job_id)
    aged = datetime.now(UTC) - timedelta(seconds=seconds)
    with sync_system_session() as session:
        session.execute(update(Job).where(Job.id == key).values(updated_at=aged))
        session.commit()


def _leased_page(
    client: TestClient, headers: dict[str, str], project: dict, *, agent_name: str = "Laptop"
) -> tuple[dict, dict]:
    """One agent holding one real page, claimed over HTTP."""
    agent = _running_agent(client, headers, agent_name)
    _prefer_local(client, headers)
    ids = _make_part(client, headers, project)
    _submit_segment(client, headers, project, ids)
    page = _claim(client, agent["device_token"]).json()["page"]
    assert page is not None
    return agent, page


# ---------------------------------------------------------------------------
# An expired lease returns the page to the queue
# ---------------------------------------------------------------------------


def test_an_expired_lease_returns_the_page_to_a_different_agent(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The whole point of the lease.

    A researcher closes their lid mid-page. The page must be picked up by the
    next agent to ask - any agent, not the one that stopped - without the
    researcher resubmitting anything.
    """
    stopped_agent, page = _leased_page(client, owner_headers, owner_project, agent_name="Lid shut")
    job_id = page["product_job_id"]
    second_agent = _running_agent(client, owner_headers, "Second laptop")
    assert _stored_job(job_id).claimed_by == agent_claim_owner(
        uuid.UUID(stopped_agent["device_id"])
    )

    # The lid closes. Nothing else happens for longer than the lease.
    _age_by(job_id, get_device_settings().device_lease_seconds + 60)

    # The next agent to ask sweeps on the way in and then claims.
    response = _claim_allowed_to_sweep(client, second_agent["device_token"])

    assert response.status_code == 200, response.text
    handed_over = response.json()["page"]
    assert handed_over is not None, "the abandoned page was not returned to the queue"
    assert handed_over["product_job_id"] == job_id
    stored = _stored_job(job_id)
    assert stored.status is JobStatus.waiting
    assert stored.claimed_by == agent_claim_owner(uuid.UUID(second_agent["device_id"]))
    # A fresh lease, and a fresh inference_job_id: the stopped agent's callback
    # contract no longer matches this row.
    assert stored.inference_job_id != uuid.UUID(page["inference_job_id"])


def test_an_expired_lease_is_re_pended_and_never_failed(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The terminal state this must not reach is ``failed``.

    The page is aged past *both* deadlines - the 240s waiting timeout and the
    600s lease - so if the two populations were ever collapsed into one the
    waiting sweep would fail it here. It must come back as ``pending`` instead,
    with no error and no completion, because nothing failed: a page was put down.
    """
    _agent, page = _leased_page(client, owner_headers, owner_project)
    job_id = page["product_job_id"]
    settings = get_job_settings()
    device_settings = get_device_settings()
    _age_by(job_id, device_settings.device_lease_seconds + 60)

    swept = run_stale_job_sweep()

    assert swept == 1
    stored = _stored_job(job_id)
    assert stored.status is JobStatus.pending
    assert stored.status is not JobStatus.failed, "an abandoned lease must never be failed"
    assert stored.error is None
    assert WAITING_TIMEOUT_ERROR not in (stored.error or "")
    assert stored.completed_at is None
    # The claim is gone, so *any* agent may take it next.
    assert stored.claimed_by is None
    assert stored.inference_job_id is None
    assert stored.started_at is None
    assert stored.heartbeat_at is None
    # And the failing sweep, run directly at the same age, still declines it.
    assert (
        fail_stale_waiting_jobs(waiting_timeout_seconds=settings.job_worker_waiting_timeout_seconds)
        == 0
    )


def test_the_stopped_agent_cannot_report_on_a_page_it_no_longer_holds(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """A laptop that wakes up hours later must not overwrite the job.

    The same guarantee ``reclaim_stale_running_jobs`` gets from ``_owned_by``:
    the claim is cleared with the lease, so the zombie's credential no longer
    names the holder of that page.
    """
    agent, page = _leased_page(client, owner_headers, owner_project)
    job_id = page["product_job_id"]
    _age_by(job_id, get_device_settings().device_lease_seconds + 60)
    assert run_stale_job_sweep() == 1

    late = client.post(
        CALLBACK_URL,
        headers=_device_headers(agent["device_token"]),
        json={
            "inference_job_id": page["inference_job_id"],
            "product_job_id": job_id,
            "task": "segment",
            "status": "done",
            "output": {"kind": "segment", "data": {"lines": []}},
        },
    )

    assert late.status_code == 403, late.text
    assert _stored_job(job_id).status is JobStatus.pending


# ---------------------------------------------------------------------------
# A page inside its lease is untouched
# ---------------------------------------------------------------------------


def test_a_page_inside_its_lease_is_not_swept(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """An agent that is merely slow keeps its page."""
    agent, page = _leased_page(client, owner_headers, owner_project)
    job_id = page["product_job_id"]
    _age_by(job_id, get_device_settings().device_lease_seconds - 60)

    assert run_stale_job_sweep() == 0

    stored = _stored_job(job_id)
    assert stored.status is JobStatus.waiting
    assert stored.claimed_by == agent_claim_owner(uuid.UUID(agent["device_id"]))
    assert stored.inference_job_id == uuid.UUID(page["inference_job_id"])


def test_a_page_past_the_waiting_timeout_but_inside_its_lease_is_untouched(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The two timeouts are distinct, and this is the window that proves it.

    At 300 seconds an agent-held page is well past the 240-second waiting
    timeout and well inside the 600-second lease. Before this issue the waiting
    sweep would have failed it here. The lease is the only deadline that governs
    an agent-held page.
    """
    settings = get_job_settings()
    device_settings = get_device_settings()
    age = (settings.job_worker_waiting_timeout_seconds + device_settings.device_lease_seconds) / 2
    assert settings.job_worker_waiting_timeout_seconds < age < device_settings.device_lease_seconds

    agent, page = _leased_page(client, owner_headers, owner_project)
    job_id = page["product_job_id"]
    _age_by(job_id, age)

    assert run_stale_job_sweep() == 0

    stored = _stored_job(job_id)
    assert stored.status is JobStatus.waiting
    assert stored.claimed_by == agent_claim_owner(uuid.UUID(agent["device_id"]))
    # No other agent may take it either: the lease has not expired.
    other = _running_agent(client, owner_headers, "Impatient laptop")
    assert _claim(client, other["device_token"]).json()["page"] is None


# ---------------------------------------------------------------------------
# The device lease is distinct from, and shorter than, the global job timeout
# ---------------------------------------------------------------------------


def test_the_device_lease_is_shorter_than_the_global_job_timeout() -> None:
    """1800 seconds is right for a server that does not sleep and wrong for a
    laptop that does. A closed lid must not hold a page for half an hour."""
    device_settings = get_device_settings()
    settings = get_job_settings()

    assert device_settings.device_lease_seconds == 600
    assert device_settings.device_lease_seconds < settings.job_worker_running_timeout_seconds
    assert settings.job_worker_running_timeout_seconds == 1800
    # And distinct from the waiting timeout, which governs the other population.
    assert device_settings.device_lease_seconds != settings.job_worker_waiting_timeout_seconds


def test_a_hosted_worker_inherits_the_same_lease(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """One lease, both credentials.

    A server that does not sleep will never trip it, but a hosted worker that is
    redeployed mid-page must not strand that page any more than a laptop does -
    and the recovery is the same sweep, not a cloud-specific one.
    """
    first = client.post(
        CLAIM_URL,
        headers={**SERVICE_HEADERS, WORKER_NAME_HEADER: "cloud-worker"},
        json={"wait_seconds": 0},
    )
    assert first.status_code == 200, first.text
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)
    page = client.post(CLAIM_URL, headers=SERVICE_HEADERS, json={"wait_seconds": 0}).json()["page"]
    assert page is not None and page["execution_target"] == "cloud"

    _age_by(job_id, get_device_settings().device_lease_seconds + 60)

    assert run_stale_job_sweep() == 1

    stored = _stored_job(job_id)
    assert stored.status is JobStatus.pending
    assert stored.status is not JobStatus.failed
    assert stored.claimed_by is None
    # And the next hosted worker to ask gets it.
    reclaimed = client.post(CLAIM_URL, headers=SERVICE_HEADERS, json={"wait_seconds": 0})
    assert reclaimed.json()["page"]["product_job_id"] == job_id


def test_a_platform_dispatched_page_still_fails_on_the_waiting_timeout() -> None:
    """Whatever is correct for a non-agent row stays correct.

    A job the platform itself dispatched and got no callback for has nothing to
    retry, so it still fails at 240 seconds. Only the ``claimed_by`` prefix
    separates the two, and both behaviours have to survive.
    """
    settings = get_job_settings()
    job_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.segment,
                status=JobStatus.waiting,
                payload={"handler": "noop"},
                inference_job_id=uuid.uuid4(),
                claimed_by="build-host:4711",
                updated_at=datetime.now(UTC) - timedelta(seconds=600),
            )
        )
        session.commit()

    assert (
        fail_stale_waiting_jobs(waiting_timeout_seconds=settings.job_worker_waiting_timeout_seconds)
        == 1
    )
    stored = _stored_job(job_id)
    assert stored.status is JobStatus.failed
    assert stored.error.startswith(WAITING_TIMEOUT_ERROR)


def test_an_unclaimed_waiting_row_is_failed_not_re_pended() -> None:
    """``NOT LIKE`` is NULL for an unclaimed row.

    Written the naive way, the exemption for agent rows would have swallowed
    every job with a NULL ``claimed_by`` - which is most of the population the
    waiting timeout exists for.
    """
    settings = get_job_settings()
    job_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.segment,
                status=JobStatus.waiting,
                payload={"handler": "noop"},
                claimed_by=None,
                updated_at=datetime.now(UTC) - timedelta(seconds=600),
            )
        )
        session.commit()

    assert release_expired_device_leases(lease_seconds=1.0) == 0
    assert (
        fail_stale_waiting_jobs(waiting_timeout_seconds=settings.job_worker_waiting_timeout_seconds)
        == 1
    )
    assert _stored_job(job_id).status is JobStatus.failed


# ---------------------------------------------------------------------------
# Concurrent sweeps do not double-release or corrupt job state
# ---------------------------------------------------------------------------


def _seed_expired_agent_pages(count: int) -> list[uuid.UUID]:
    """``count`` pages held by a stopped agent, each already past its lease."""
    device_id = uuid.uuid4()
    aged = datetime.now(UTC) - timedelta(seconds=get_device_settings().device_lease_seconds + 60)
    job_ids = [uuid.uuid4() for _ in range(count)]
    with sync_system_session() as session:
        for job_id in job_ids:
            session.add(
                Job(
                    id=job_id,
                    type=JobType.segment,
                    status=JobStatus.waiting,
                    payload={"handler": "noop"},
                    inference_job_id=uuid.uuid4(),
                    claimed_by=agent_claim_owner(device_id),
                    started_at=aged,
                    heartbeat_at=aged,
                    updated_at=aged,
                )
            )
        session.commit()
    return job_ids


def test_concurrent_lease_releases_never_double_release_a_page() -> None:
    """Eight sweepers, one queue, real Postgres.

    The advisory lock in ``run_stale_job_sweep`` normally serializes replicas, so
    this deliberately goes underneath it and calls the release directly from
    eight threads at once. If ``FOR UPDATE SKIP LOCKED`` and the repeated
    ``status``/``claimed_by`` predicates were not doing their job, the counts
    would sum to more than the number of pages.
    """
    job_ids = _seed_expired_agent_pages(12)

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(
            pool.map(
                lambda _: release_expired_device_leases(
                    lease_seconds=get_device_settings().device_lease_seconds
                ),
                range(8),
            )
        )

    # Every page released exactly once, across all eight sweepers.
    assert sum(counts) == len(job_ids), f"pages were released more than once: {counts}"
    with sync_system_session() as session:
        rows = session.execute(select(Job).where(Job.id.in_(job_ids))).scalars().all()
    assert len(rows) == len(job_ids)
    for row in rows:
        assert row.status is JobStatus.pending
        assert row.claimed_by is None
        assert row.inference_job_id is None
        assert row.error is None
        assert row.completed_at is None


def test_concurrent_agents_racing_a_swept_queue_each_get_a_distinct_page(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The recovery path end to end, under real concurrent HTTP.

    Six pages abandoned by one stopped agent, two live agents polling at once.
    Every page must end up with exactly one of them - a re-pended page is an
    ordinary queue entry and inherits ``FOR UPDATE SKIP LOCKED`` unchanged.
    """
    stopped, first_page = _leased_page(client, owner_headers, owner_project, agent_name="Stopped")
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    job_ids = [first_page["product_job_id"]]
    for _ in range(5):
        job_id = _submit_segment(client, owner_headers, owner_project, ids)
        response = _claim(client, stopped["device_token"])
        assert response.json()["page"]["product_job_id"] == job_id
        job_ids.append(job_id)

    lease = get_device_settings().device_lease_seconds
    for job_id in job_ids:
        _age_by(job_id, lease + 60)

    alpha = _running_agent(client, owner_headers, "Alpha")
    beta = _running_agent(client, owner_headers, "Beta")
    tokens = [alpha["device_token"], beta["device_token"]] * 4

    reset_stale_sweep_throttle()
    with ThreadPoolExecutor(max_workers=len(tokens)) as pool:
        responses = list(pool.map(lambda token: _claim(client, token), tokens))

    assert all(response.status_code == 200 for response in responses)
    handed = [
        response.json()["page"]["product_job_id"]
        for response in responses
        if response.json()["page"] is not None
    ]
    assert len(handed) == len(set(handed)), "the same page was handed to two agents"
    with sync_system_session() as session:
        rows = (
            session.execute(select(Job).where(Job.id.in_([uuid.UUID(j) for j in job_ids])))
            .scalars()
            .all()
        )
    # Nothing failed, and nothing is still held by the agent that stopped.
    stopped_owner = agent_claim_owner(uuid.UUID(stopped["device_id"]))
    for row in rows:
        assert row.status in (JobStatus.pending, JobStatus.waiting)
        assert row.status is not JobStatus.failed
        assert row.claimed_by != stopped_owner
