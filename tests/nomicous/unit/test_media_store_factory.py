"""Media store factory selection."""

from backend.document.infrastructure.media_store import LocalMediaStore, get_media_store


def test_get_media_store_defaults_to_local() -> None:
    store = get_media_store()
    assert isinstance(store, LocalMediaStore)
