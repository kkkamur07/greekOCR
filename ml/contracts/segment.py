"""Segment task request/response contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ml.contracts.common import ImageBytes


class SegmentGeometryKind(str, Enum):
    polygon = "polygon"
    rectangle = "rectangle"


class SegmentRunRequest(BaseModel):
    registry_model_id: str = Field(min_length=1)
    registry_tag: str = Field(default="stable", min_length=1)
    image_bytes: ImageBytes
    params: dict[str, Any] = Field(default_factory=dict)


class SegmentBlock(BaseModel):
    external_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    box: dict[str, Any]


class SegmentLine(BaseModel):
    external_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    block_external_id: str | None = None
    baseline: dict[str, Any]
    mask: dict[str, Any] | None = None
    kind: SegmentGeometryKind = SegmentGeometryKind.polygon
    points: list[list[float]] = Field(min_length=4)
    kraken_ceiling: list[list[float]] | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("points", "kraken_ceiling")
    @classmethod
    def validate_points(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if any(len(point) != 2 for point in value):
            raise ValueError("each point must contain x and y")
        return value


class SegmentRunResponse(BaseModel):
    blocks: list[SegmentBlock] = Field(default_factory=list)
    lines: list[SegmentLine] = Field(default_factory=list)
