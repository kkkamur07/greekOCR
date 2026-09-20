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
from datetime import datetime
from types import ModuleType, SimpleNamespace

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
        assert ref.split("://", 1)[1].split("?", 1)[0] == registry_id
        assert display_name != registry_id


def test_candidate_names_cover_old_and_new_rows(seed_module: ModuleType) -> None:
    assert seed_module.candidate_names_for("blla-segment") == ("blla-segment", "kraken")
    assert seed_module.candidate_names_for("ppocr-segment") == ("ppocr-segment", "ppocr")
    assert seed_module.candidate_names_for("greek-calamari-v1") == ("greek-calamari-v1",)


def test_duplicate_name_rows_keep_the_oldest(seed_module: ModuleType) -> None:
    old = SimpleNamespace(name="blla-segment", created_at=datetime(2026, 1, 1))
    new = SimpleNamespace(name="kraken", created_at=datetime(2026, 2, 1))
    keep, delete = seed_module.split_duplicate_models([new, old], "ppocr")
    assert keep is old
    assert delete == [new]


def test_default_segment_ids_offer_both_models(
    seed_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEFAULT_SEGMENT_MODEL", raising=False)
    reloaded = importlib.reload(seed_module)
    assert set(reloaded.SEGMENT_MODELS) == {"blla-segment", "ppocr-segment"}


def test_duplicate_artifact_ref_rows_converge_to_display_name(
    seed_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two rows sharing one artifact_ref (old and new name) seed to one row."""
    import asyncio

    from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask

    ref = seed_module.artifact_ref_for("blla-segment")

    class FakeScalars:
        def __init__(self, rows: list) -> None:
            self._rows = rows

        def all(self) -> list:
            return list(self._rows)

    class FakeResult:
        def __init__(self, rows: list) -> None:
            self._rows = rows

        def scalars(self) -> FakeScalars:
            return FakeScalars(self._rows)

    class FakeSession:
        def __init__(self, rows: list) -> None:
            self._rows = list(rows)
            self.deleted = []

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def execute(self, stmt: object) -> FakeResult:
            return FakeResult(self._rows)

        async def delete(self, obj: object) -> None:
            self.deleted.append(obj)
            if obj in self._rows:
                self._rows.remove(obj)

        async def flush(self) -> None:
            return None

        async def commit(self) -> None:
            return None

        async def refresh(self, obj: object) -> None:
            return None

    old = InferenceModel(
        name="blla-segment",
        provider="kraken",
        task=InferenceTask.segment,
        artifact_ref=ref,
        default_params={"device": "cpu"},
        created_at=datetime(2026, 1, 1),
    )
    new = InferenceModel(
        name="kraken",
        provider="kraken",
        task=InferenceTask.segment,
        artifact_ref=ref,
        default_params={"device": "cpu"},
        created_at=datetime(2026, 2, 1),
    )
    session = FakeSession([old, new])
    monkeypatch.setattr(seed_module, "system_session", lambda: session)

    model = asyncio.run(seed_module._upsert_model(name="blla-segment", task=InferenceTask.segment))

    assert model.name == "kraken"
    assert model.artifact_ref == ref
    assert session.deleted == [old]
    assert session._rows == [model]
    assert len(session._rows) == 1
