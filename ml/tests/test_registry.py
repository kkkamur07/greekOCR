"""registry.yaml validation and weight path helpers."""

import pytest

from ml.contracts import ComputeDevice, MLTask, RegistryArchitecture
from ml.registry import get_model_entry, load_registry
from ml.weights import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_WEIGHTS_ROOT,
    bundled_weights_dir,
    resolve_weights_source,
    weight_cache_dir,
)


def test_registry_yaml_validates_model_entries():
    registry = load_registry()

    calamari = get_model_entry(registry, "greek-calamariv1", "stable")
    assert calamari.task == MLTask.transcribe
    assert calamari.architecture == RegistryArchitecture.calamari
    assert calamari.device == ComputeDevice.cpu
    assert calamari.versions["stable"].weights_source.startswith("file://")

    kraken = get_model_entry(registry, "kraken-blla", "stable")
    assert kraken.task == MLTask.segment
    assert kraken.architecture == RegistryArchitecture.kraken_segment
    assert kraken.device == ComputeDevice.cpu


def test_registry_weights_source_resolves_under_ml_root():
    registry = load_registry()
    uri = registry.models["greek-calamariv1"].versions["stable"].weights_source
    path = resolve_weights_source(uri)

    assert path == (
        DEFAULT_WEIGHTS_ROOT / "calamari" / "greek-calamariv1" / "stable.ckpt"
    )


def test_weights_source_rejects_paths_outside_ml_root():
    with pytest.raises(ValueError, match="relative to ML_ROOT"):
        resolve_weights_source("file:///etc/passwd")

    with pytest.raises(ValueError, match="within ML_ROOT"):
        resolve_weights_source("file://../pyproject.toml")


def test_bundled_weights_dir_resolves_from_registry_source():
    registry = load_registry()
    uri = registry.models["greek-calamariv1"].versions["stable"].weights_source

    assert (
        bundled_weights_dir(uri)
        == DEFAULT_WEIGHTS_ROOT / "calamari" / "greek-calamariv1"
    )


def test_weight_cache_layout():
    cache_dir = weight_cache_dir("greek-calamariv1", "stable")
    assert cache_dir == DEFAULT_CACHE_ROOT / "greek-calamariv1" / "stable"
