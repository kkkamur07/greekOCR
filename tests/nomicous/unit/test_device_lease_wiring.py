"""The lease's structural guarantees: opportunistic, and no new machinery.

The behaviour of the sweep is proved live, against Postgres and the real app, in
``tests/nomicous/integration/test_device_lease.py``. What is left here is the set
of claims a live test cannot make, because they are claims about what does *not*
exist: no background worker, no release endpoint, no second timeout on the
agent-held population.

These are source and SQL assertions, not substitutes for a database. Nothing here
stands in for Postgres; there is nothing to stand in for.
"""

from __future__ import annotations

import ast
import inspect
import os

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")

from sqlalchemy.dialects import postgresql

from backend.core.app import create_app
from backend.core.settings.device import DeviceSettings
from backend.core.settings.job import JobSettings
from backend.jobs.infrastructure import job_repository, stale_sweep, worker
from backend.jobs.infrastructure.job_repository import (
    AGENT_CLAIM_PREFIX,
    _held_by_agent,
    _not_held_by_agent,
)

# Every long-lived task the application already starts. The lease adds none.
_PRE_EXISTING_BACKGROUND_TASKS = {
    "platform_job_notification_loop",
    "worker_loop",
    "media_gc_loop",
}


def _created_task_names(module) -> set[str]:
    """Names passed to ``asyncio.create_task`` anywhere in *module*'s source."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "create_task" or not node.args:
            continue
        started = node.args[0]
        if isinstance(started, ast.Call) and isinstance(started.func, ast.Name):
            names.add(started.func.id)
    return names


# ---------------------------------------------------------------------------
# No background process was added
# ---------------------------------------------------------------------------


def test_no_background_process_was_added_for_the_lease() -> None:
    """The deployment is serverless and must stay that way.

    A lease reaper is the obvious design and the wrong one here: there is no
    process to run it on the production API, which is why ADR 0005 promises
    abandonment is "the existing stale sweep" rather than a new mechanism. If
    someone adds a fourth task to the lifespan, this names it.
    """
    from backend.core import app as app_module

    assert _created_task_names(app_module) == _PRE_EXISTING_BACKGROUND_TASKS


def test_the_lease_sweep_spawns_nothing_of_its_own() -> None:
    """No thread, no timer, no scheduler, no loop of its own.

    ``asyncio.to_thread`` is allowed and required - the sweeps use sync sessions
    and must not run on the event loop - but it is a call the caller awaits, not
    a process that outlives the request.
    """
    for module in (job_repository, stale_sweep, worker):
        source = inspect.getsource(module)
        assert "threading.Thread" not in source
        assert "Timer(" not in source
        assert "apscheduler" not in source
        assert "celery" not in source

    sweep_source = inspect.getsource(stale_sweep)
    assert "asyncio.create_task" not in sweep_source
    assert "while True" not in sweep_source
    assert "asyncio.to_thread(run_stale_job_sweep)" in sweep_source


def test_the_lease_is_released_only_from_paths_that_already_run() -> None:
    """The two callers are a read path and the existing worker tick.

    Both already existed; neither is new work on a schedule.
    """
    callers = {
        name
        for name, module in (
            ("run_stale_job_sweep", stale_sweep),
            ("process_one_job", worker),
        )
        if "release_expired_device_leases" in inspect.getsource(getattr(module, name))
    }

    assert callers == {"run_stale_job_sweep", "process_one_job"}


def test_no_release_or_heartbeat_endpoint_was_added() -> None:
    """The lease is recovered by expiry, not by anyone calling to say so.

    A killed process cannot call a release endpoint, and a slept laptop cannot
    heartbeat. Both would be liveness mechanisms for a window nothing runs past.
    """
    paths = set(create_app().openapi()["paths"])

    assert not [path for path in paths if "heartbeat" in path or "release" in path]
    assert {path for path in paths if path.startswith("/device/v1/jobs")} == {
        "/device/v1/jobs/claim"
    }


# ---------------------------------------------------------------------------
# One lease, not one of two timeouts
# ---------------------------------------------------------------------------


def test_the_device_lease_is_shorter_than_the_global_job_timeout(monkeypatch) -> None:
    """1800 seconds is right for a server that does not sleep and wrong for a
    laptop that does."""
    for name in ("DEVICE_LEASE_SECONDS", "JOB_WORKER_RUNNING_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)

    assert DeviceSettings().device_lease_seconds == 600
    assert JobSettings().job_worker_running_timeout_seconds == 1800
    assert DeviceSettings().device_lease_seconds < JobSettings().job_worker_running_timeout_seconds


def test_a_hosted_worker_inherits_the_same_lease() -> None:
    """One lease for both credentials (ADR 0003): a laptop and a hosted worker
    are the same kind of thing here. A server that does not sleep will never trip
    it, but it is now the one timeout rather than one of two - so the release
    discriminates on nothing but "an agent holds this, and the lease is up".

    The live half of this is in the integration suite, where a hosted worker's
    abandoned cloud page is re-pended by the same sweep.
    """
    source = inspect.getsource(job_repository.release_expired_device_leases)

    assert "execution_target" not in source
    assert "user_id" not in source
    assert "is_service_worker" not in source
    assert "_held_by_agent()" in source


def test_the_two_waiting_populations_are_complements_over_one_column() -> None:
    """The waiting sweep and the lease sweep must partition ``waiting`` exactly.

    An overlap fails a page that should have been re-queued; a gap leaves a row
    that no sweep ever touches. Both predicates read the same column, and the
    complement names NULL explicitly because ``NOT LIKE`` is NULL for an
    unclaimed row - which would have exempted most of the waiting population.
    """
    held = str(_held_by_agent().compile(dialect=postgresql.dialect()))
    not_held = str(_not_held_by_agent().compile(dialect=postgresql.dialect()))

    assert "claimed_by LIKE" in held
    assert "claimed_by IS NULL" in not_held
    assert "claimed_by NOT LIKE" in not_held


def test_the_claim_writes_the_prefix_the_sweep_reads() -> None:
    """One constant, read from one place.

    Two copies of ``"agent:"`` is a silent partition failure: the claim writes
    one string and the sweep looks for the other, so no lease is ever recovered
    and every abandoned page is failed by the waiting timeout instead.
    """
    import uuid

    from backend.jobs.application import job_claim_service

    assert job_claim_service.AGENT_CLAIM_PREFIX is AGENT_CLAIM_PREFIX
    assert job_claim_service.agent_claim_owner(uuid.uuid4()).startswith(AGENT_CLAIM_PREFIX)
    assert "agent:" not in inspect.getsource(job_claim_service.agent_claim_owner)
