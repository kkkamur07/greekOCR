"""Data layout — directories exist under configured data root."""

from annote.services.data_layout import ensure_data_directories, export_dir, list_required_subdirs


def test_ensure_data_directories_creates_expected_layout(data_root):
    """Startup ensures all PRD data subdirectories exist."""
    ensure_data_directories(data_root)

    for subdir in list_required_subdirs():
        assert (data_root / subdir).is_dir()
    assert export_dir(data_root).name == "export"
