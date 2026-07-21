"""Model, dataset, and collection publish coverage without live Hub access."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.hf.publish import (
    DatasetStagingRef,
    MockPublishClient,
    ModelStagingRef,
    build_model_card,
    build_dataset_readme,
    dataset_staging_dir,
    hub_repo_slug,
    model_staging_dir,
    plan_collection_sync,
    plan_dataset_publish,
    plan_model_publish,
    publish_dataset,
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


def _seed_dataset_staging(staging_root: Path) -> Path:
    ref = DatasetStagingRef(dataset_slug="greek-manuscript-lines")
    staging_dir = dataset_staging_dir(ref, staging_root=staging_root)
    images_dir = staging_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "line-0001.png").write_bytes(b"mock-line-crop")
    (staging_dir / "labels.csv").write_text(
        "image,transcription\nimages/line-0001.png,λόγος\n",
        encoding="utf-8",
    )
    return staging_dir


def test_blla_uses_script_agnostic_segmentation_repo_slug():
    assert hub_repo_slug("segmentation", "blla") == "segmentation-blla"
    assert hub_repo_slug("syriac", "calamari") == "syriac-htr-calamari"
    card = build_model_card(
        ModelStagingRef("segmentation", "blla", "v1", "stable"),
        namespace="nomicous",
        task="segment",
        registry_model_id="blla-segment",
    )
    assert "language:" not in card
    assert "# Document Segmentation (blla)" in card


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


def test_publish_dataset_creates_dataset_repo_and_materializes_pairing_readme(
    tmp_path: Path,
):
    _seed_dataset_staging(tmp_path)
    client = MockPublishClient()

    plan = plan_dataset_publish(
        dataset_slug="greek-manuscript-lines",
        namespace="nomicous",
        script="greek",
        staging_root=tmp_path,
        dry_run=False,
    )
    workspace = tmp_path / "dataset-workspace"
    publish_dataset(plan, publish_client=client, workspace=workspace)

    assert plan.repo_id == "nomicous/greek-manuscript-lines"
    assert client.repos == [("nomicous/greek-manuscript-lines", "dataset", False)]
    assert client.uploads[0][1:3] == ("nomicous/greek-manuscript-lines", "dataset")
    readme = (workspace / "README.md").read_text(encoding="utf-8")
    assert "## Pairing convention" in readme
    assert "future **registry model ids**" in readme
    assert (workspace / "images" / "line-0001.png").read_bytes() == b"mock-line-crop"


def test_dataset_publish_dry_run_does_not_call_hub_client(tmp_path: Path):
    _seed_dataset_staging(tmp_path)
    client = MockPublishClient()
    plan = plan_dataset_publish(
        dataset_slug="greek-manuscript-lines",
        namespace="nomicous",
        script="greek",
        staging_root=tmp_path,
    )

    publish_dataset(plan, publish_client=client)

    assert client.repos == []
    assert client.uploads == []


@pytest.mark.parametrize(
    ("dataset_slug", "script", "error"),
    [
        ("greek-htr-calamari", "greek", "dataset slug must follow"),
        ("syriac-manuscript-lines", "greek", "dataset slug must follow"),
        ("greek-uncial-htr-lines", "Greek", "script must be a lowercase slug"),
    ],
)
def test_dataset_publish_rejects_slug_outside_context_convention(
    tmp_path: Path, dataset_slug: str, script: str, error: str
):
    with pytest.raises(ValueError, match=error):
        plan_dataset_publish(
            dataset_slug=dataset_slug,
            namespace="nomicous",
            script=script,
            staging_root=tmp_path,
        )


def test_dataset_publish_rejects_unpaired_crop(tmp_path: Path):
    staging_dir = _seed_dataset_staging(tmp_path)
    (staging_dir / "images" / "line-0002.png").write_bytes(b"unpaired")

    with pytest.raises(ValueError, match="without labels.csv pairings"):
        plan_dataset_publish(
            dataset_slug="greek-manuscript-lines",
            namespace="nomicous",
            script="greek",
            staging_root=tmp_path,
        )


def test_dataset_readme_distinguishes_dataset_from_model_repo():
    readme = build_dataset_readme(
        DatasetStagingRef(dataset_slug="syriac-manuscript-lines"),
        namespace="nomicous",
        script="syriac",
    )

    assert "Hub dataset repo" in readme
    assert "separate **Hub model repos**" in readme
    assert "relative path below `images/`" in readme


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
