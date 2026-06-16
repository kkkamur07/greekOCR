"""Canonical segment DTOs shared by adapters and merge services."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from backend.document.infrastructure.orm_models import LineGeometryKind


class CanonicalBlock(BaseModel):
    external_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    box: dict


class CanonicalLine(BaseModel):
    external_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    block_external_id: str | None = None
    baseline: dict
    mask: dict | None = None
    kind: LineGeometryKind = LineGeometryKind.polygon
    points: list[list[float]] = Field(min_length=4)
    kraken_ceiling: list[list[float]] | None = None
    source_metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("points", "kraken_ceiling")
    @classmethod
    def validate_points(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if any(len(point) != 2 for point in value):
            raise ValueError("each point must contain x and y")
        return value


class CanonicalSegmentResult(BaseModel):
    blocks: list[CanonicalBlock] = Field(default_factory=list)
    lines: list[CanonicalLine] = Field(default_factory=list)
