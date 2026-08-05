"""Shared HTTP mapping for synchronous /inference/v1/run failures."""

from __future__ import annotations

from fastapi import HTTPException, status

from inference.admission import CLIENT_INPUT_ERROR
from src.hf.resolve.artifacts import ArtifactIntegrityError

UNKNOWN_MODEL_ERROR = "Unknown registry model or tag"
WEIGHTS_UNAVAILABLE_ERROR = "Model weights are not available"
WEIGHTS_INTEGRITY_ERROR = "Model weights failed integrity verification"
RUNTIME_UNAVAILABLE_ERROR = "Inference runtime is unavailable for this model"
INTERNAL_RUN_ERROR = "Internal inference error"


def http_exception_for_run_error(exc: Exception) -> HTTPException:
    """Map a runner failure to a stable client-visible HTTPException.

    ``ArtifactIntegrityError`` subclasses ``ValueError`` and must be checked
    first so integrity failures stay service errors (503), not client 422s.
    """
    if isinstance(exc, ArtifactIntegrityError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=WEIGHTS_INTEGRITY_ERROR,
        )
    if isinstance(exc, KeyError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=UNKNOWN_MODEL_ERROR,
        )
    if isinstance(exc, FileNotFoundError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=WEIGHTS_UNAVAILABLE_ERROR,
        )
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=CLIENT_INPUT_ERROR,
        )
    if isinstance(exc, RuntimeError):
        # BLLAUnavailableError and CalamariUnavailableError both subclass
        # RuntimeError: the artifact or runtime is broken, not the client
        # request.
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=RUNTIME_UNAVAILABLE_ERROR,
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=INTERNAL_RUN_ERROR,
    )
