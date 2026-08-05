"""Local-run provenance: the job id the caller sees is the one on the lines.

Lives in the inference test tree because it belongs to the local (browser
orchestrated) inference slice; the platform-side integration coverage for the
same endpoints is in ``tests/nomicous/integration/test_local_inference_persist``.
No database: the merge and the job repository are both stubbed, because what is
under test is which id flows where.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from backend.document.application import local_inference_service as service_module
from backend.document.application.local_inference_service import LocalInferenceService
from inference.contracts.segment import SegmentRunResponse


class _FakeJob:
    def __init__(self, result: dict) -> None:
        self.id = uuid4()
        self.result = result


class _FakeJobRepository:
    """Stand-in for the async repository; records what was written."""

    recorded: list[_FakeJob] = []

    def __init__(self, _session: object) -> None:
        pass

    async def record_local_job(self, **kwargs: object) -> _FakeJob:
        job = _FakeJob(dict(kwargs["result"]))  # type: ignore[arg-type]
        _FakeJobRepository.recorded.append(job)
        return job


class _FakeSession:
    def __init__(self) -> None:
        self.deleted: list[object] = []
        self.commits = 0

    async def delete(self, instance: object) -> None:
        self.deleted.append(instance)

    async def commit(self) -> None:
        self.commits += 1


def _segment_output() -> SegmentRunResponse:
    return SegmentRunResponse(
        blocks=[],
        lines=[
            {
                "external_id": "l-1",
                "order": 0,
                "baseline": {"points": [[1.0, 1.0], [2.0, 1.0]]},
                "points": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]],
            }
        ],
    )


@pytest.fixture
def stubbed_segment_merge(monkeypatch: pytest.MonkeyPatch):
    """Replace everything around the merge and capture the stamped job id."""
    _FakeJobRepository.recorded = []
    stamped: list[UUID] = []

    class _FakeMergeService:
        def apply_sync(self, _session, *, part_id, canonical_segment, job_id, commit):
            stamped.append(job_id)
            if _FakeMergeService.fails:
                raise RuntimeError("merge exploded")
            return SimpleNamespace(
                blocks_count=len(canonical_segment.blocks),
                lines_count=len(canonical_segment.lines),
                added_lines=1,
                pruned_lines=0,
                preserved_manual_lines=0,
            )

    _FakeMergeService.fails = False

    @contextmanager
    def _sync_session():
        yield object()

    monkeypatch.setattr(service_module, "JobRepository", _FakeJobRepository)
    monkeypatch.setattr(service_module, "SegmentMergeService", _FakeMergeService)
    monkeypatch.setattr(service_module, "sync_system_session", _sync_session)
    for name in ("_require_member", "_load_document_in_project", "_document_part_or_404"):
        monkeypatch.setattr(
            LocalInferenceService,
            name,
            lambda *_args, **_kwargs: _awaited(object()),
            raising=True,
        )
    return SimpleNamespace(stamped=stamped, merge=_FakeMergeService)


async def _awaited(value: object) -> object:
    return value


async def _persist_segment(session: _FakeSession) -> dict:
    return await LocalInferenceService().persist_local_segment(
        session,  # type: ignore[arg-type]
        SimpleNamespace(id=uuid4()),  # type: ignore[arg-type]
        uuid4(),
        uuid4(),
        uuid4(),
        registry_model_id="blla-segment",
        registry_tag="stable",
        output=_segment_output(),
    )


async def test_returned_job_id_is_the_one_stamped_on_the_merged_lines(
    stubbed_segment_merge,
) -> None:
    session = _FakeSession()

    result = await _persist_segment(session)

    assert len(stubbed_segment_merge.stamped) == 1
    # The id in the response is the durable job row's id, and it is the same id
    # the merge wrote into every line's source_metadata - otherwise nothing that
    # a local run produced could ever be traced back to the job the user saw.
    assert result["job_id"] == str(stubbed_segment_merge.stamped[0])
    assert result["job_id"] == str(_FakeJobRepository.recorded[0].id)
    # The summary still lands on the persisted job, not just in the response.
    assert _FakeJobRepository.recorded[0].result["lines_count"] == 1


async def test_failed_merge_leaves_no_done_job_row_behind(stubbed_segment_merge) -> None:
    stubbed_segment_merge.merge.fails = True
    session = _FakeSession()

    with pytest.raises(RuntimeError, match="merge exploded"):
        await _persist_segment(session)

    assert session.deleted == [_FakeJobRepository.recorded[0]]
