"""The abandoned-direct-upload sweep, against live Postgres.

``begin_upload`` commits a part row before the browser PUTs any bytes, so a
closed tab leaves a pending row behind - and possibly a blob referenced by
nothing. The sweep's population is decided by a ``WHERE`` clause over the
sentinel prefix and ``created_at``, so it is proven here, where there are rows,
not with a mocked session that has no ``WHERE`` clause.

**Expiry is created by writing the timestamp, not by waiting for it.** Same rule
as ``test_job_worker_sweeps.py``: the shortest deadline in play is three hours,
so a test that waited for one is a test nobody runs.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.document.infrastructure.media_gc import sweep_abandoned_part_uploads
from backend.document.infrastructure.orm_models import DocumentPart, MediaDeletionIntent
from infrastructure.db import sync_system_session
from tests.nomicous.integration.helpers import make_part

pytestmark = pytest.mark.integration


def test_sweep_reaps_only_expired_pending_rows(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    slug = f"proj-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/projects", headers=auth_headers, json={"slug": slug, "name": "Sweep project"}
    )
    assert created.status_code == 201, created.text
    project = created.json()
    document_id, finalized_part_id = make_part(client, auth_headers, project)

    expired = datetime.now(UTC) - timedelta(hours=4)
    abandoned_id = uuid.uuid4()
    bare_id = uuid.uuid4()
    fresh_id = uuid.uuid4()
    minted_key = f"parts/{abandoned_id}/scan.png"
    with sync_system_session() as session:
        session.add_all(
            [
                # A begin whose browser never finalized, past the presign window.
                DocumentPart(
                    id=abandoned_id,
                    document_id=uuid.UUID(document_id),
                    order=10,
                    image_key=f"pending:{minted_key}",
                    created_at=expired,
                ),
                # A row from before the sentinel carried the minted key: reap the
                # row, but there is no key to queue a deletion for.
                DocumentPart(
                    id=bare_id,
                    document_id=uuid.UUID(document_id),
                    order=11,
                    image_key="pending",
                    created_at=expired,
                ),
                # An upload still inside its window must be left to finish.
                DocumentPart(
                    id=fresh_id,
                    document_id=uuid.UUID(document_id),
                    order=12,
                    image_key=f"pending:parts/{fresh_id}.webp",
                ),
            ]
        )
        # Age the finalized part too: the sweep must filter on the sentinel,
        # not on age alone.
        finalized = session.get(DocumentPart, uuid.UUID(finalized_part_id))
        assert finalized is not None
        finalized.created_at = expired
        session.commit()

    assert sweep_abandoned_part_uploads() == 2

    with sync_system_session() as session:
        remaining = set(
            session.execute(
                select(DocumentPart.id).where(DocumentPart.document_id == uuid.UUID(document_id))
            ).scalars()
        )
        queued_keys = list(session.execute(select(MediaDeletionIntent.image_key)).scalars())

    assert remaining == {uuid.UUID(finalized_part_id), fresh_id}
    assert minted_key in queued_keys
    assert "pending" not in queued_keys

    # A second pass finds nothing: the sweep converges.
    assert sweep_abandoned_part_uploads() == 0
