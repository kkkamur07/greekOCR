"""Webhook auth for internal ML callbacks."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status
from ml.contracts.webhooks import ML_WEBHOOK_SECRET_HEADER

from backend.core.settings.ml import get_ml_settings


def require_ml_webhook_secret(
    x_ml_webhook_secret: str | None = Header(default=None, alias=ML_WEBHOOK_SECRET_HEADER),
) -> None:
    configured = get_ml_settings().ml_webhook_secret
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML webhook secret is not configured",
        )
    if x_ml_webhook_secret is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing ML webhook secret",
        )
    if not secrets.compare_digest(x_ml_webhook_secret, configured):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid ML webhook secret",
        )
