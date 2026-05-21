"""Application settings — split by concern, single .env at backend/core/.env."""

from backend.core.settings.app import AppSettings, get_app_settings
from backend.core.settings.auth import AuthSettings, get_auth_settings
from backend.core.settings.infrastructure import InfrastructureSettings, get_infrastructure_settings
from backend.core.settings.model import ModelSettings, get_model_settings

__all__ = [
    "AppSettings",
    "AuthSettings",
    "InfrastructureSettings",
    "ModelSettings",
    "get_app_settings",
    "get_auth_settings",
    "get_infrastructure_settings",
    "get_model_settings",
]
