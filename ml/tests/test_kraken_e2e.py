"""Opt-in Kraken segmentation E2E through the ML job queue."""

from __future__ import annotations

import base64
import os
import shutil
from importlib import resources
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from ml.api.app import create_app
from ml.contracts.common import MLJobStatus
from ml.infrastructure.job_repository import get_job_by_id
from ml.jobs.worker import process_next_job
from ml.registry import get_model_entry, load_registry
from ml.weights import resolve_weights_source


def _seed_bundled_kraken_weights(weights_path: Path) -> None:
    """Copy Kraken's bundled BLLA model to the registry path for local E2E runs."""
    try:
        source = resources.files("kraken").joinpath("blla.mlmodel")
    except ModuleNotFoundError:
        pytest.skip("Kraken is not installed; run with the kraken extra")

    if not source.is_file():
        pytest.skip("Kraken package does not include bundled blla.mlmodel")

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with resources.as_file(source) as source_path:
        shutil.copyfile(source_path, weights_path)


@pytest.mark.integration
def test_segment_job_queue_with_real_kraken_weights() -> None:
    registry_model_id = os.environ.get("KRAKEN_TEST_MODEL_ID", "kraken-blla")
    registry_tag = os.environ.get("KRAKEN_TEST_REGISTRY_TAG", "stable")
    image_path = Path(os.environ.get("KRAKEN_TEST_IMAGE_PATH", ""))

    registry = load_registry()
    entry = get_model_entry(registry, registry_model_id, registry_tag)
    weights_path = resolve_weights_source(entry.versions[registry_tag].weights_source)

    if not weights_path.is_file():
        _seed_bundled_kraken_weights(weights_path)
    if not image_path.is_file():
        pytest.skip("set KRAKEN_TEST_IMAGE_PATH to a real page image")

    client = TestClient(create_app())
    response = client.post(
        "/ml/v1/jobs",
        json={
            "task": "segment",
            "registry_model_id": registry_model_id,
            "registry_tag": registry_tag,
            "product_job_id": str(uuid4()),
            "image_bytes": base64.b64encode(image_path.read_bytes()).decode(),
        },
    )
    assert response.status_code == 201
    ml_job_id = UUID(response.json()["ml_job_id"])

    assert process_next_job() is True

    job = get_job_by_id(ml_job_id)
    assert job is not None
    assert job.status == MLJobStatus.done
    assert job.output is not None
    assert job.output["lines"]
    assert all(line["source_metadata"]["adapter"] == "kraken" for line in job.output["lines"])
