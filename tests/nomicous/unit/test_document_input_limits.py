"""Bounds on untrusted document input and on the unauthenticated read surface.

The local-inference persist routes accept model output straight from the browser and the
public layout endpoint answers anonymous callers, so both need the caps their siblings
(``PUT /lines`` and every other list endpoint) already enforce.
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
    MAX_LINE_TEXT_CHARS,
    MAX_LOCAL_TRANSCRIBE_LINES,
    MAX_PUBLIC_LAYOUT_LINES,
    MAX_REPLACE_PART_LINES,
    BlockPatchRequest,
    LineCreateRequest,
    LinePatchRequest,
    LinesReplaceRequest,
    LineUpsertRequest,
    LocalSegmentPersistRequest,
    LocalTranscribePersistRequest,
    SegmentPartRequest,
)
from backend.document.infrastructure.orm_models import Line


def _points(count: int) -> list[list[float]]:
    return [[float(index), float(index)] for index in range(count)]


def _line_payload(points: int) -> dict:
    return {"external_id": "l1", "order": 0, "baseline": {}, "points": _points(points)}


# --- Geometry bound on the ordinary line routes ---


def test_create_line_rejects_unbounded_point_lists() -> None:
    LineCreateRequest(order=0, points=_points(MAX_LINE_GEOMETRY_POINTS))

    with pytest.raises(PydanticValidationError):
        LineCreateRequest(order=0, points=_points(MAX_LINE_GEOMETRY_POINTS + 1))


def test_patch_line_rejects_unbounded_point_lists() -> None:
    LinePatchRequest(points=_points(MAX_LINE_GEOMETRY_POINTS))

    with pytest.raises(PydanticValidationError):
        LinePatchRequest(points=_points(MAX_LINE_GEOMETRY_POINTS + 1))


def test_bulk_replace_rejects_unbounded_point_lists() -> None:
    def upsert(count: int) -> dict:
        return {"order": 0, "points": _points(count)}

    LinesReplaceRequest(lines=[upsert(MAX_LINE_GEOMETRY_POINTS)])

    with pytest.raises(PydanticValidationError):
        LinesReplaceRequest(lines=[upsert(MAX_LINE_GEOMETRY_POINTS + 1)])


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


def test_segmentation_cannot_request_more_points_than_the_platform_stores() -> None:
    SegmentPartRequest(target_max_points=MAX_LINE_GEOMETRY_POINTS)

    with pytest.raises(PydanticValidationError):
        SegmentPartRequest(target_max_points=MAX_LINE_GEOMETRY_POINTS + 1)


def test_block_patch_still_accepts_a_partial_update() -> None:
    assert BlockPatchRequest(order=2).model_dump(exclude_unset=True) == {"order": 2}


# --- Local transcribe persist ---


def _transcribe_line(text: str = "alpha") -> dict:
    return {"line_id": str(uuid.uuid4()), "text": text, "confidence": 0.5}


def test_local_transcribe_caps_line_count() -> None:
    assert MAX_LOCAL_TRANSCRIBE_LINES == MAX_REPLACE_PART_LINES

    with pytest.raises(PydanticValidationError):
        LocalTranscribePersistRequest(
            registry_model_id="m",
            lines=[_transcribe_line() for _ in range(MAX_LOCAL_TRANSCRIBE_LINES + 1)],
        )


def test_local_transcribe_caps_per_line_text() -> None:
    with pytest.raises(PydanticValidationError):
        LocalTranscribePersistRequest(
            registry_model_id="m",
            lines=[_transcribe_line("x" * (MAX_LINE_TEXT_CHARS + 1))],
        )


def test_local_transcribe_caps_total_text() -> None:
    oversized = [_transcribe_line("y" * MAX_LINE_TEXT_CHARS) for _ in range(200)]

    with pytest.raises(PydanticValidationError):
        LocalTranscribePersistRequest(registry_model_id="m", lines=oversized)


def test_local_transcribe_rejects_misaligned_character_confidences() -> None:
    short = _transcribe_line("ab")
    short["character_confidences"] = [{"char": "a", "confidence": 0.5}]

    with pytest.raises(PydanticValidationError):
        LocalTranscribePersistRequest(registry_model_id="m", lines=[short])

    mismatched = _transcribe_line("ab")
    mismatched["character_confidences"] = [
        {"char": "a", "confidence": 0.5},
        {"char": "z", "confidence": 0.5},
    ]

    with pytest.raises(PydanticValidationError):
        LocalTranscribePersistRequest(registry_model_id="m", lines=[mismatched])


def test_local_transcribe_accepts_a_normal_payload() -> None:
    payload = _transcribe_line("ab")
    payload["character_confidences"] = [
        {"char": "a", "confidence": 0.5},
        {"char": "b", "confidence": 0.25},
    ]

    request = LocalTranscribePersistRequest(registry_model_id="m", lines=[payload])

    assert request.lines[0].character_confidences is not None


# --- Local segment persist ---


def test_local_segment_caps_line_count() -> None:
    with pytest.raises(PydanticValidationError):
        LocalSegmentPersistRequest(
            registry_model_id="m",
            output={"lines": [_line_payload(4) for _ in range(MAX_REPLACE_PART_LINES + 1)]},
        )


def test_local_segment_caps_points_per_line() -> None:
    with pytest.raises(PydanticValidationError):
        LocalSegmentPersistRequest(
            registry_model_id="m",
            output={"lines": [_line_payload(MAX_LINE_GEOMETRY_POINTS + 1)]},
        )


def test_local_segment_accepts_a_normal_payload() -> None:
    request = LocalSegmentPersistRequest(
        registry_model_id="m",
        output={
            "blocks": [{"external_id": "b1", "order": 0, "box": {"x": 1}}],
            "lines": [_line_payload(8)],
        },
    )

    assert len(request.output.lines[0].points) == 8


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
        return [], rows[:limit]


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
    assert fake.calls[0]["limit"] == 3  # limit + 1 probe row
    resumed = decode_cursor(response.next_cursor)
    assert resumed.id == lines[1].id


@pytest.mark.asyncio
async def test_public_layout_resumes_from_the_cursor(monkeypatch) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    lines = [_line(base + timedelta(seconds=index)) for index in range(5)]
    fake = _FakeLayoutService(lines)
    monkeypatch.setattr(public_api, "_service", fake)

    first = await public_api.get_published_layout(
        uuid.uuid4(), uuid.uuid4(), db=None, limit=2, cursor=None
    )
    second = await public_api.get_published_layout(
        uuid.uuid4(), uuid.uuid4(), db=None, limit=2, cursor=first.next_cursor
    )

    assert [line.id for line in second.lines] == [lines[2].id, lines[3].id]
    assert second.next_cursor is not None


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
