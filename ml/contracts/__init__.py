"""Portable ML task contracts shared by ml/api and annote HTTP client."""

from ml.contracts.common import (
    ComputeDevice,
    MLJobStatus,
    MLTask,
    RegistryArchitecture,
)
from ml.contracts.jobs import JobCallbackRequest, JobSubmitRequest, JobSubmitResponse
from ml.contracts.segment import (
    SegmentBlock,
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
    "JobCallbackRequest",
    "JobSubmitRequest",
    "JobSubmitResponse",
    "MLJobStatus",
    "MLTask",
    "RegistryArchitecture",
    "SegmentBlock",
    "SegmentLine",
    "SegmentRunRequest",
    "SegmentRunResponse",
    "TranscribeRunRequest",
    "TranscribeRunResponse",
]
