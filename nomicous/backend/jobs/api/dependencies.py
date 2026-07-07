"""Webhook auth for internal inference callbacks."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status
from inference.contracts.webhooks import INFERENCE_WEBHOOK_SECRET_HEADER

from backend.core.settings.ml import get_inference_settings

__all__ = ["INFERENCE_WEBHOOK_SECRET_HEADER", "require_inference_webhook_secret"]


def require_inference_webhook_secret(
    x_inference_webhook_secret: str | None = Header(default=None, alias=INFERENCE_WEBHOOK_SECRET_HEADER),
) -> None:
    configured = get_inference_settings().inference_webhook_secret
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference webhook secret is not configured",
        )
    if x_inference_webhook_secret is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing inference webhook secret",
        )
    if not secrets.compare_digest(x_inference_webhook_secret, configured):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid inference webhook secret",
        )
