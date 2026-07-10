"""Shared helpers for nomicous integration tests."""

from __future__ import annotations

import time
from functools import lru_cache

from fastapi.testclient import TestClient

from tests.fixtures.paths import MINIMAL_PNG

__all__ = ["MINIMAL_PNG", "documents_url", "poll_job", "stored_minimal_page_bytes"]


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
