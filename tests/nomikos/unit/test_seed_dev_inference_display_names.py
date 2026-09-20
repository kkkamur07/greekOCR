"""Display names for the dev seed of segment catalog rows.

The pickers render ``inference_models.name`` while dispatch parses the
registry id out of ``artifact_ref``, so a segment row is shown as ``kraken``
or ``ppocr`` but must still point at its registry id. These tests cover the
pure name/ref mapping; the database upsert itself is exercised by running the
script, so no database is needed here.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest

from tests.fixtures.paths import REPO_ROOT

PLATFORM_SCRIPTS = REPO_ROOT / "scripts" / "platform"


@pytest.fixture(scope="module")
def seed_module() -> Iterator[ModuleType]:
    """Import the standalone seed script; it resolves `_bootstrap` as a sibling."""
    sys.path.insert(0, str(PLATFORM_SCRIPTS))
    try:
        yield importlib.import_module("seed_dev_inference")
    finally:
        sys.path.remove(str(PLATFORM_SCRIPTS))


def test_segment_display_names(seed_module: ModuleType) -> None:
    assert seed_module.display_name_for("blla-segment") == "kraken"
    assert seed_module.display_name_for("ppocr-segment") == "ppocr"


def test_unknown_ids_keep_their_registry_id_as_name(seed_module: ModuleType) -> None:
    assert seed_module.display_name_for("greek-calamari-v1") == "greek-calamari-v1"


def test_artifact_refs_use_registry_ids_never_display_names(
    seed_module: ModuleType,
) -> None:
    assert seed_module.artifact_ref_for("blla-segment") == "registry://blla-segment?tag=stable"
    assert seed_module.artifact_ref_for("ppocr-segment") == "registry://ppocr-segment?tag=stable"
    for registry_id, display_name in seed_module.SEGMENT_DISPLAY_NAMES.items():
        ref = seed_module.artifact_ref_for(registry_id)
        assert seed_module.registry_id_from_artifact_ref(ref) == registry_id
        assert ref.split("://", 1)[1].split("?", 1)[0] == registry_id
        assert display_name != registry_id


def test_candidate_names_cover_old_and_new_rows(seed_module: ModuleType) -> None:
    assert seed_module.candidate_names_for("blla-segment") == ("blla-segment", "kraken")
    assert seed_module.candidate_names_for("ppocr-segment") == ("ppocr-segment", "ppocr")
    assert seed_module.candidate_names_for("greek-calamari-v1") == ("greek-calamari-v1",)


def test_default_segment_ids_offer_both_models(
    seed_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEFAULT_SEGMENT_MODEL", raising=False)
    reloaded = importlib.reload(seed_module)
    assert set(reloaded.SEGMENT_MODELS) == {"blla-segment", "ppocr-segment"}
