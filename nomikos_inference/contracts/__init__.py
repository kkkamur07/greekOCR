"""Portable ML task contracts shared by inference and annote HTTP client."""

from nomikos_inference.contracts.common import (
    ComputeDevice,
    HostEligibility,
    ImageBytes,
    InferenceJobStatus,
    InferenceTask,
    RegistryArchitecture,
)
from nomikos_inference.contracts.jobs import (
    JobCallbackRequest,
    JobOutput,
    JobSubmitRequest,
    SegmentJobOutput,
    TranscribeJobOutput,
)
from nomikos_inference.contracts.segment import (
    SegmentBlock,
    SegmentGeometryKind,
    SegmentLine,
    SegmentRunResponse,
)
from nomikos_inference.contracts.transcribe import (
    TRANSCRIBE_LINE_ERROR,
    CharacterConfidence,
    TranscribeBatchLineResult,
    TranscribeBatchRunResponse,
    TranscribeLineRegion,
    TranscribeRunResponse,
)

__all__ = [
    "TRANSCRIBE_LINE_ERROR",
    "CharacterConfidence",
    "ComputeDevice",
    "HostEligibility",
    "ImageBytes",
    "JobCallbackRequest",
    "JobOutput",
    "JobSubmitRequest",
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
