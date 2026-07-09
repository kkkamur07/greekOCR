"""Essential model publish and collection sync coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.hf.publish import (
    MockPublishClient,
    ModelStagingRef,
    model_staging_dir,
    plan_collection_sync,
    plan_model_publish,
    publish_model,
    sync_collection,
)


def _seed_model_staging(staging_root: Path) -> None:
    ref = ModelStagingRef(
        script="greek",
        architecture="calamari",
        model_version="v1",
        registry_tag="stable",
    )
    staging_dir = model_staging_dir(ref, staging_root=staging_root)
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "best.pt").write_bytes(b"mock-checkpoint")


def test_publish_model_uploads_and_tags_revision(tmp_path: Path):
    _seed_model_staging(tmp_path)
    client = MockPublishClient(upload_revision="sha-publish")

    plan = plan_model_publish(
        script="greek",
        architecture="calamari",
        model_version="v1",
        registry_tag="stable",
        namespace="nomicous",
        task="transcribe",
        staging_root=tmp_path,
        dry_run=False,
    )
    publish_model(plan, publish_client=client, workspace=tmp_path / "workspace")

    assert plan.repo_id == "nomicous/greek-htr-calamari"
    assert plan.weights_source == "hf://nomicous/greek-htr-calamari@stable"
    assert client.tags == [("nomicous/greek-htr-calamari", "stable", "sha-publish", "model")]


def test_sync_collection_updates_hub_membership(tmp_path: Path):
    collection_path = tmp_path / "collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "namespace": "nomicous",
                "slug": "nomos",
                "title": "Nomicous Manuscript HTR",
                "description": "Test collection",
                "hub_slug": "nomicous/nomos-test",
                "models": [{"slug": "greek-htr-calamari", "note": "Greek v1"}],
                "datasets": [],
            }
        ),
        encoding="utf-8",
    )
    client = MockPublishClient()
    plan = plan_collection_sync(collection_path=collection_path, dry_run=False)

    sync_collection(plan, publish_client=client)

    assert ("nomicous/nomos-test", "nomicous/greek-htr-calamari", "model", "Greek v1") in (
        client.collection_items
    )
