"""Shared helpers for nomicous integration tests."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.fixtures.paths import MINIMAL_PNG

__all__ = [
    "MINIMAL_PNG",
    "assert_api_error",
    "documents_url",
    "pair_inference_device",
    "poll_job",
    "stored_minimal_page_bytes",
    "user_id_for_email",
]


def user_id_for_email(email: str) -> uuid.UUID:
    from backend.users.infrastructure.orm_models import User
    from infrastructure.db import sync_system_session

    with sync_system_session() as session:
        return session.execute(select(User.id).where(User.email == email)).scalar_one()


def pair_inference_device(
    *,
    user_id: uuid.UUID,
    host: str = "cloud",
    seen_seconds_ago: float | None = 5,
) -> uuid.UUID:
    """Give an **inference host** **capacity** by writing a recently-seen device.

    Submission is gated on capacity, so any test that expects a 202 has to say
    which host is running. This writes the real row the real query reads -
    ``last_seen_at`` is the production signal, and controlling it is how capacity
    is made deterministic without patching a clock.

    ``seen_seconds_ago=None`` writes a device that has never checked in: paired
    but not running, which must not count as capacity.
    """
    from backend.ml.domain.execution import ExecutionTarget
    from backend.ml.infrastructure.device_orm_models import HelperDevice
    from infrastructure.db import sync_system_session

    device_id = uuid.uuid4()
    now = datetime.now(UTC)
    with sync_system_session() as session:
        session.add(
            HelperDevice(
                id=device_id,
                user_id=user_id,
                inference_host=ExecutionTarget(host),
                name=f"{host} worker",
                platform="linux-x86_64",
                helper_version="0.2.0",
                capabilities={},
                token_hash="a" * 64,
                token_prefix="nmd1.test",
                last_seen_at=(
                    None if seen_seconds_ago is None else now - timedelta(seconds=seen_seconds_ago)
                ),
            )
        )
        session.commit()
    return device_id


def assert_api_error(
    response,
    *,
    code: str,
    message: str | None = None,
) -> dict:
    """Assert allowlisted API error shape (code/message); allow correlation `ref`."""
    body = response.json()
    assert "error" in body
    error = body["error"]
    assert error["code"] == code
    if message is not None:
        assert error["message"] == message
    if "ref" in error:
        assert isinstance(error["ref"], str) and error["ref"]
    return error


@lru_cache
def stored_minimal_page_bytes() -> bytes:
    """Bytes expected after upload normalization to stored WebP."""
    from backend.document.infrastructure.media_store.encoding import encode_part_image

    return encode_part_image(MINIMAL_PNG)


def documents_url(project_id: str) -> str:
    return f"/projects/{project_id}/documents"


def poll_job(
    client: TestClient,
    job_id: str,
    *,
    expect_status: str = "done",
    headers: dict[str, str],
    timeout: float = 5.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        body = response.json()
        if body["status"] == expect_status:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach status {expect_status!r} in {timeout}s")
