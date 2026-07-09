"""Service-to-service auth for the inference API."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status
from inference.contracts.webhooks import INFERENCE_SERVICE_SECRET_HEADER

from inference.infrastructure.settings import get_inference_settings

__all__ = ["INFERENCE_SERVICE_SECRET_HEADER", "require_inference_service_secret"]


def require_inference_service_secret(
    x_inference_service_secret: str | None = Header(
        default=None, alias=INFERENCE_SERVICE_SECRET_HEADER
    ),
) -> None:
    configured = get_inference_settings().inference_service_secret
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service secret is not configured",
        )
    if x_inference_service_secret is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing inference service secret",
        )
    if not secrets.compare_digest(x_inference_service_secret, configured):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid inference service secret",
        )
