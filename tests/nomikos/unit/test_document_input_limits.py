"""Bounds on untrusted document input and on the unauthenticated read surface.

The public layout endpoint answers anonymous callers, so it needs the caps its
siblings (``PUT /lines`` and every other list endpoint) already enforce.

The local-inference persist routes were the other subject here. They existed
because the browser ran the model itself and posted the result back; ADR 0002
retired that path (#60), and an **inference agent** now reports through the same
job callback a hosted worker uses, so its output is bounded where every other
job result is.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError as PydanticValidationError

from backend.core.api.pagination import decode_cursor
from backend.document.api import public as public_api
from backend.document.api.schemas import (
    DEFAULT_PUBLIC_LAYOUT_LINES,
    MAX_LINE_GEOMETRY_POINTS,
    MAX_LINE_IDS_PER_REQUEST,
    MAX_LINE_TEXT_CHARS,
    MAX_PART_IDS_PER_REQUEST,
    MAX_PUBLIC_LAYOUT_LINES,
    CopyToGroundTruthRequest,
    LayoutResetRequest,
    LineCreateRequest,
    LineTranscriptionPatchRequest,
    LineUpsertRequest,
    ReorderPartsRequest,
    TranscribePartRequest,
)
from backend.document.application.document_catalog import PublicLayoutPage
from backend.document.infrastructure.orm_models import Line


def _points(count: int) -> list[list[float]]:
    return [[float(index), float(index)] for index in range(count)]


def _line_payload(points: int) -> dict:
    return {"external_id": "l1", "order": 0, "baseline": {}, "points": _points(points)}


# --- Text and id-list bounds ---
# Every other text field in the module carries a cap and every other list a length;
# these five did not, in a module that bounds things deliberately everywhere else.


def test_line_transcription_patch_bounds_its_text() -> None:
    """The only uncapped text field in the module, over an unbounded ``Text`` column."""
    LineTranscriptionPatchRequest(text="a" * MAX_LINE_TEXT_CHARS)

    with pytest.raises(PydanticValidationError):
        LineTranscriptionPatchRequest(text="a" * (MAX_LINE_TEXT_CHARS + 1))


@pytest.mark.parametrize(
    "model",
    [LayoutResetRequest, TranscribePartRequest, CopyToGroundTruthRequest],
    ids=lambda cls: cls.__name__,
)
def test_line_id_lists_are_bounded(model) -> None:
    """`CopyToGroundTruthRequest.line_ids` reaches `.in_(...)` unexamined."""
    model(line_ids=[uuid.uuid4() for _ in range(4)])

    with pytest.raises(PydanticValidationError):
        model(line_ids=[uuid.uuid4()] * (MAX_LINE_IDS_PER_REQUEST + 1))


def test_reorder_part_ids_are_bounded() -> None:
    """One UPDATE per element, so the length of this list is a work budget."""
    ReorderPartsRequest(part_ids=[uuid.uuid4(), uuid.uuid4()])

    with pytest.raises(PydanticValidationError):
        ReorderPartsRequest(part_ids=[uuid.uuid4()] * (MAX_PART_IDS_PER_REQUEST + 1))

    with pytest.raises(PydanticValidationError):
        ReorderPartsRequest(part_ids=[])


# --- Geometry bound on the ordinary line routes ---


def test_create_line_rejects_unbounded_point_lists() -> None:
    LineCreateRequest(order=0, points=_points(MAX_LINE_GEOMETRY_POINTS))

    with pytest.raises(PydanticValidationError):
        LineCreateRequest(order=0, points=_points(MAX_LINE_GEOMETRY_POINTS + 1))


def test_upsert_rejects_unbounded_kraken_ceiling() -> None:
    with pytest.raises(PydanticValidationError):
        LineUpsertRequest(
            order=0,
            points=_points(4),
            kraken_ceiling=_points(MAX_LINE_GEOMETRY_POINTS + 1),
        )


def test_point_pair_validation_still_applies() -> None:
    with pytest.raises(PydanticValidationError):
        LineCreateRequest(order=0, points=[[0, 0], [1, 1], [2, 2], [3, 3, 3]])


# --- Public layout read surface ---


def _line(created_at: datetime) -> Line:
    line = Line(
        id=uuid.uuid4(),
        part_id=uuid.uuid4(),
        order=0,
        baseline={},
        points=[[0.0, 0.0], [1.0, 1.0]],
    )
    line.created_at = created_at
    line.transcriptions = []
    return line


class _FakeLayoutService:
    def __init__(self, lines: list[Line]) -> None:
        self.lines = lines
        self.calls: list[dict] = []

    async def list_document_layout_public(
        self, _session, _project_id, _document_id, *, limit, cursor=None
    ):
        self.calls.append({"limit": limit, "cursor": cursor})
        rows = self.lines
        if cursor is not None:
            rows = [
                line for line in rows if (line.created_at, line.id) > (cursor.created_at, cursor.id)
            ]
        # Mirrors the catalog contract: it probes one row past the page size itself.
        return PublicLayoutPage(blocks=[], blocks_truncated=False, lines=rows[: limit + 1])


@pytest.mark.asyncio
async def test_public_layout_truncates_and_emits_a_cursor(monkeypatch) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    lines = [_line(base + timedelta(seconds=index)) for index in range(5)]
    fake = _FakeLayoutService(lines)
    monkeypatch.setattr(public_api, "_service", fake)

    response = await public_api.get_published_layout(
        uuid.uuid4(), uuid.uuid4(), db=None, limit=2, cursor=None
    )

    assert len(response.lines) == 2
    assert response.next_cursor is not None
    assert fake.calls[0]["limit"] == 2  # the page size; the catalog probes past it
    resumed = decode_cursor(response.next_cursor)
    assert resumed.id == lines[1].id


@pytest.mark.asyncio
async def test_public_layout_last_page_has_no_cursor(monkeypatch) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    lines = [_line(base + timedelta(seconds=index)) for index in range(2)]
    fake = _FakeLayoutService(lines)
    monkeypatch.setattr(public_api, "_service", fake)

    response = await public_api.get_published_layout(
        uuid.uuid4(), uuid.uuid4(), db=None, limit=10, cursor=None
    )

    assert len(response.lines) == 2
    assert response.next_cursor is None


def test_public_layout_limit_is_bounded_by_the_route_signature() -> None:
    from fastapi.routing import APIRoute

    route = next(
        route
        for route in public_api.router.routes
        if isinstance(route, APIRoute) and route.path.endswith("/layout")
    )
    limit = next(param for param in route.dependant.query_params if param.name == "limit")
    metadata = limit.field_info

    assert metadata.default == DEFAULT_PUBLIC_LAYOUT_LINES
    assert any(getattr(item, "le", None) == MAX_PUBLIC_LAYOUT_LINES for item in metadata.metadata)
