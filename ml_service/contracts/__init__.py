"""Portable ML task contracts shared by ml_service and annote HTTP client."""

from ml_service.contracts.common import (
    ComputeDevice,
    ImageBytes,
    MLJobStatus,
    MLTask,
    RegistryArchitecture,
)
from ml_service.contracts.jobs import (
    JobCallbackRequest,
    JobOutput,
    JobSubmitRequest,
    JobSubmitResponse,
    SegmentJobOutput,
    TranscribeJobOutput,
)
from ml_service.contracts.segment import (
    SegmentBlock,
    SegmentGeometryKind,
    SegmentLine,
    SegmentRunRequest,
    SegmentRunResponse,
)
from ml_service.contracts.transcribe import (
    CharacterConfidence,
    TranscribeRunRequest,
    TranscribeRunResponse,
)

__all__ = [
    "CharacterConfidence",
    "ComputeDevice",
    "ImageBytes",
    "JobCallbackRequest",
    "JobOutput",
    "JobSubmitRequest",
    "JobSubmitResponse",
    "MLJobStatus",
    "MLTask",
    "RegistryArchitecture",
    "SegmentBlock",
    "SegmentGeometryKind",
    "SegmentJobOutput",
    "SegmentLine",
    "SegmentRunRequest",
    "SegmentRunResponse",
    "TranscribeRunRequest",
    "TranscribeRunResponse",
    "TranscribeJobOutput",
]
