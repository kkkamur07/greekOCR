"""Portable ML task contracts shared by ml/api and annote HTTP client."""

from ml.contracts.common import (
    ComputeDevice,
    ImageBytes,
    MLJobStatus,
    MLTask,
    RegistryArchitecture,
)
from ml.contracts.jobs import (
    JobCallbackRequest,
    JobOutput,
    JobSubmitRequest,
    JobSubmitResponse,
    SegmentJobOutput,
    TranscribeJobOutput,
)
from ml.contracts.segment import (
    SegmentBlock,
    SegmentGeometryKind,
    SegmentLine,
    SegmentRunRequest,
    SegmentRunResponse,
)
from ml.contracts.transcribe import (
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
