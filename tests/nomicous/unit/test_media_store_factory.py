"""Media store factory selection."""

from backend.core.settings.storage import get_storage_settings
from backend.document.infrastructure.media_store import LocalMediaStore, get_media_store


def test_get_media_store_defaults_to_local(monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    get_storage_settings.cache_clear()
    get_media_store.cache_clear()
    try:
        assert isinstance(get_media_store(), LocalMediaStore)
    finally:
        get_media_store.cache_clear()
        get_storage_settings.cache_clear()
