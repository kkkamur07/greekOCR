"""Object storage settings (local filesystem or Supabase Storage)."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._cache import settings_cache
from backend.core.settings._env import env_settings_config

StorageBackend = Literal["local", "supabase"]


class StorageSettings(BaseSettings):
    model_config = env_settings_config()

    storage_backend: StorageBackend = Field(default="local", alias="STORAGE_BACKEND")
    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_service_role_key: str | None = Field(default=None, alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_storage_bucket: str = Field(default="document-media", alias="SUPABASE_STORAGE_BUCKET")
    media_webp_lossless: bool = Field(default=True, alias="MEDIA_WEBP_LOSSLESS")
    media_webp_quality: int = Field(default=95, alias="MEDIA_WEBP_QUALITY")
    media_url_signing_secret: str | None = Field(
        default=None,
        alias="MEDIA_URL_SIGNING_SECRET",
        description=(
            "Keys the short-lived signed links the *local* media store hands an inference "
            "agent. Unused when STORAGE_BACKEND=supabase, which signs its own. Falls back "
            "to JWT_SECRET; see url_signing_key()."
        ),
    )

    def url_signing_key(self) -> str:
        """Key for a signed page image link.

        Unlike ``DEVICE_TOKEN_HMAC_SECRET`` - which gets a dedicated secret and a
        production hard stop, because rotating it unpairs every UI-less laptop
        for 180 days - falling back to ``JWT_SECRET`` here is cheap: the longest
        thing a rotation can break is a link that had under a minute left to
        live, and the agent simply claims again. So this is a dial for a
        deployment that wants media signing separated from session signing, not a
        credential the platform refuses to start without.
        """
        if self.media_url_signing_secret:
            return self.media_url_signing_secret
        from backend.core.settings.auth import get_auth_settings

        return get_auth_settings().jwt_secret


@settings_cache
def get_storage_settings() -> StorageSettings:
    return StorageSettings()
