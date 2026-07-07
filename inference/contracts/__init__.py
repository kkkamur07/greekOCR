"""Portable ML task contracts shared by inference and annote HTTP client."""

from inference.contracts.common import (
    ComputeDevice,
    ImageBytes,
    InferenceJobStatus,
    InferenceTask,
    RegistryArchitecture,
)
from inference.contracts.jobs import (
    JobCallbackRequest,
    JobOutput,
    JobSubmitRequest,
    JobSubmitResponse,
    SegmentJobOutput,
    TranscribeJobOutput,
)
from inference.contracts.segment import (
    SegmentBlock,
    SegmentGeometryKind,
    SegmentLine,
    SegmentRunResponse,
)
from inference.contracts.transcribe import (
    CharacterConfidence,
    TranscribeBatchLineResult,
    TranscribeBatchRunResponse,
    TranscribeLineRegion,
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
    "InferenceJobStatus",
    "InferenceTask",
    "RegistryArchitecture",
    "SegmentBlock",
    "SegmentGeometryKind",
    "SegmentJobOutput",
    "SegmentLine",
    "SegmentRunResponse",
    "TranscribeBatchLineResult",
    "TranscribeBatchRunResponse",
    "TranscribeJobOutput",
    "TranscribeLineRegion",
    "TranscribeRunResponse",
]
