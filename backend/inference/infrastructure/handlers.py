"""Test job handlers — noop sleep and intentional failure."""

from __future__ import annotations

import time

from backend.inference.infrastructure.orm_models import Job


class TestJobHandlerError(Exception):
    """Raised by the fail test handler to exercise error persistence."""


def run_test_handler(job: Job) -> dict:
    handler = (job.payload or {}).get("handler", "noop")
    if handler == "fail":
        raise TestJobHandlerError("intentional test failure")
    time.sleep(0.1)
    return {"ok": True}
