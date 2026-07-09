"""Pydantic DTOs for documents and parts."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from backend.document.infrastructure.orm_models import (
    DocumentWorkflow,
    LineGeometryKind,
    LineSource,
    TranscriptionKind,
)

MAX_PAGE_TRANSCRIPTION_CHARS = 1_000_000
MAX_PAGE_TRANSCRIPTION_LINES = 10_000
MAX_REPLACE_PART_LINES = 10_000


class DocumentCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=512)


class DocumentUpdateRequest(BaseModel):
    name: str | SkipJsonSchema[None] = Field(default=None, min_length=1, max_length=512)
    workflow: DocumentWorkflow | SkipJsonSchema[None] = None

    @field_validator("name", "workflow", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("must not be null")
        return value


class DocumentResponse(BaseModel):
    id: UUID
    project_id: UUID
    name: str
    workflow: DocumentWorkflow
    part_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentPageResponse(BaseModel):
    items: list[DocumentResponse]
    next_cursor: str | None = None


class DocumentPartResponse(BaseModel):
    id: UUID
    document_id: UUID
    order: int
    image_url: str
    width: int | None
    height: int | None
    reviewed: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentPartUpdateRequest(BaseModel):
    reviewed: bool

    @field_validator("reviewed", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("must not be null")
        return value


class DocumentWithPartsResponse(DocumentResponse):
    parts: list[DocumentPartResponse]


class ReorderPartsRequest(BaseModel):
    part_ids: list[UUID] = Field(min_length=1)


class TranscriptionLayerResponse(BaseModel):
    id: UUID
    document_id: UUID
    name: str
    kind: TranscriptionKind
    created_by_job_id: UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}


class LineTranscriptionResponse(BaseModel):
    id: UUID
    transcription_id: UUID
    transcription_kind: TranscriptionKind
    text: str
    confidence: float | None


class LineResponse(BaseModel):
    id: UUID
    part_id: UUID
    block_id: UUID | None
    order: int
    baseline: dict
    mask: dict | None
    kind: LineGeometryKind
    points: list[list[float]]
    source: LineSource
    source_metadata: dict[str, object] | None
    kraken_ceiling: list[list[float]] | None
    manual_geometry: bool
    line_transcriptions: list[LineTranscriptionResponse] = Field(default_factory=list)
    created_at: datetime


class LineUpsertRequest(BaseModel):
    id: UUID | None = None
    order: int = Field(ge=0)
    kind: LineGeometryKind = LineGeometryKind.polygon
    points: list[list[float]] = Field(min_length=4)
    block_id: UUID | None = None
    source: LineSource = LineSource.manual
    source_metadata: dict[str, object] | None = None
    kraken_ceiling: list[list[float]] | None = None
    baseline: dict | None = None
    mask: dict | None = None
    approved_text: str | None = None

    @field_validator("points", "kraken_ceiling")
    @classmethod
    def validate_points(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if any(len(point) != 2 for point in value):
            raise ValueError("each point must contain x and y")
        return value


class LinesReplaceRequest(BaseModel):
    lines: list[LineUpsertRequest] = Field(default_factory=list, max_length=MAX_REPLACE_PART_LINES)


class BlockResponse(BaseModel):
    id: UUID
    part_id: UUID
    order: int
    box: dict
    manual_geometry: bool
    created_at: datetime


class BlockCreateRequest(BaseModel):
    order: int = Field(ge=0)
    box: dict


class BlockPatchRequest(BaseModel):
    order: int | None = Field(default=None, ge=0)
    box: dict | None = None


class LineCreateRequest(BaseModel):
    order: int = Field(ge=0)
    kind: LineGeometryKind = LineGeometryKind.polygon
    points: list[list[float]] = Field(min_length=4)
    block_id: UUID | None = None
    baseline: dict | None = None
    mask: dict | None = None

    @field_validator("points")
    @classmethod
    def validate_points(cls, value: list[list[float]]) -> list[list[float]]:
        if any(len(point) != 2 for point in value):
            raise ValueError("each point must contain x and y")
        return value


class LinePatchRequest(BaseModel):
    order: int | None = Field(default=None, ge=0)
    block_id: UUID | None = None
    baseline: dict | None = None
    mask: dict | None = None
    points: list[list[float]] | None = None

    @field_validator("points")
    @classmethod
    def validate_points(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if any(len(point) != 2 for point in value):
            raise ValueError("each point must contain x and y")
        return value


class LayoutResetRequest(BaseModel):
    line_ids: list[UUID] | None = None


class TranscribePartRequest(BaseModel):
    model_id: UUID | None = None
    line_ids: list[UUID] | None = None


class SegmentPartRequest(BaseModel):
    model_id: UUID | None = None
    use_otsu_refinement: bool = False
    otsu_sphere_radius: float = Field(default=4.0, gt=0, le=128)
    target_max_points: int = Field(default=80, gt=3, le=500)
    min_iou: float = Field(default=0.97, gt=0, le=1)
    min_area_ratio: float = Field(default=0.95, gt=0, le=2)
    split_large_lines: bool = True
    split_vertical_gap_px: float = Field(default=12.0, gt=0, le=256)


class LayoutResponse(BaseModel):
    blocks: list[BlockResponse]
    lines: list[LineResponse]


class PageTranscriptionImportRequest(BaseModel):
    text: str = Field(max_length=MAX_PAGE_TRANSCRIPTION_CHARS)

    @field_validator("text")
    @classmethod
    def validate_line_count(cls, value: str) -> str:
        line_count = sum(1 for line in value.splitlines() if line.strip())
        if line_count > MAX_PAGE_TRANSCRIPTION_LINES:
            raise ValueError(f"text cannot exceed {MAX_PAGE_TRANSCRIPTION_LINES} non-empty lines")
        return value


class PageTranscriptionTextLineResponse(BaseModel):
    order: int
    text: str
    paired_line_id: UUID | None


class PairingProgressResponse(BaseModel):
    paired_lines: int
    total_lines: int
    percent: int


class PagePairingResponse(BaseModel):
    text_lines: list[PageTranscriptionTextLineResponse]
    pairing_progress: PairingProgressResponse


class ExportWarningsResponse(BaseModel):
    unpaired_segments: list[int]
    unused_text_lines: list[int]


class ExportArtifactResponse(BaseModel):
    line_id: UUID
    segment_number: int
    image_filename: str
    transcription_filename: str
    transcription_text: str
    image_base64: str


class ExportResponse(BaseModel):
    exported_count: int
    artifacts: list[ExportArtifactResponse]
    warnings: ExportWarningsResponse
    steps: list[str]


class PairTextLineRequest(BaseModel):
    line_id: UUID
    text_line_order: int = Field(ge=0)


class CopyToGroundTruthRequest(BaseModel):
    line_ids: list[UUID] | None = None


class CopyToGroundTruthResponse(BaseModel):
    copied_line_ids: list[UUID]


class LineTranscriptionPatchRequest(BaseModel):
    text: str


class LocalTranscribeLinePersistRequest(BaseModel):
    line_id: UUID
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    character_confidences: list[dict[str, object]] | None = None


class LocalTranscribePersistRequest(BaseModel):
    registry_model_id: str = Field(min_length=1)
    registry_tag: str = Field(default="stable", min_length=1)
    lines: list[LocalTranscribeLinePersistRequest] = Field(min_length=1)


class LocalTranscribePersistResponse(BaseModel):
    job_id: UUID
    transcription_id: UUID
    lines: list[dict[str, object]]


class LocalSegmentPersistRequest(BaseModel):
    registry_model_id: str = Field(min_length=1)
    registry_tag: str = Field(default="stable", min_length=1)
    output: dict[str, object]


class LocalSegmentPersistResponse(BaseModel):
    job_id: UUID
    blocks_count: int
    lines_count: int
    added_lines: int
    pruned_lines: int
    preserved_manual_lines: int


class PublicBlockResponse(BaseModel):
    id: UUID
    part_id: UUID
    order: int
    box: dict


class PublicLineResponse(BaseModel):
    id: UUID
    part_id: UUID
    order: int
    points: list[list[float]]
    line_transcriptions: list[LineTranscriptionResponse] = Field(default_factory=list)


class PublicLayoutResponse(BaseModel):
    blocks: list[PublicBlockResponse] = Field(default_factory=list)
    lines: list[PublicLineResponse] = Field(default_factory=list)


class PublicTranscriptionLayerResponse(BaseModel):
    id: UUID
    name: str
    kind: TranscriptionKind

    model_config = {"from_attributes": True}
