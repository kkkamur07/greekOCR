"""registry.yaml validation and weight path helpers."""

from pathlib import Path

import pytest
from inference.contracts import ComputeDevice, InferenceTask, RegistryArchitecture
from inference.registry import DEFAULT_REGISTRY_PATH, get_model_entry, load_registry
from inference.weights import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_WEIGHTS_ROOT,
    resolve_weights_source,
)


def _bundled_weights_dir(
    weights_source: str,
    *,
    inference_root: Path | None = None,
) -> Path:
    from inference.weights import INFERENCE_ROOT

    return resolve_weights_source(
        weights_source,
        inference_root=inference_root or INFERENCE_ROOT,
    ).parent


def _weight_cache_dir(
    registry_model_id: str,
    registry_tag: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    return cache_root / registry_model_id / registry_tag


# --- registry.yaml entries ---
# Tests bundled model metadata loads correctly. Does not run inference.


def test_registry_yaml_validates_model_entries():
    registry = load_registry()

    syriac = get_model_entry(registry, "syriac-calamariv1", "stable")
    assert syriac.task == InferenceTask.transcribe
    assert syriac.architecture == RegistryArchitecture.calamari
    assert syriac.device == ComputeDevice.cpu
    assert syriac.versions["stable"].weights_source.startswith("file://")

    calamari = get_model_entry(registry, "greek-calamariv1", "stable")
    assert calamari.task == InferenceTask.transcribe
    assert calamari.architecture == RegistryArchitecture.calamari
    assert calamari.device == ComputeDevice.cpu
    assert calamari.versions["stable"].weights_source.startswith("file://")

    kraken = get_model_entry(registry, "kraken-blla", "stable")
    assert kraken.task == InferenceTask.segment
    assert kraken.architecture == RegistryArchitecture.kraken_segment
    assert kraken.device == ComputeDevice.cpu
    assert kraken.versions["stable"].weights_source == "package://kraken/blla.mlmodel"


# --- Weight path resolution ---
# Tests file:// and package:// URIs resolve to real paths. Does not download remote weights.


def test_registry_weights_source_resolves_under_ml_root():
    registry = load_registry()
    uri = registry.models["syriac-calamariv1"].versions["stable"].weights_source
    path = resolve_weights_source(uri)

    assert path == (DEFAULT_REGISTRY_PATH.parent / "weights/calamari/syriac-calamariv1/best.ckpt")


def test_registry_package_weights_source_resolves_package_resource():
    registry = load_registry()
    uri = registry.models["kraken-blla"].versions["stable"].weights_source
    path = resolve_weights_source(uri)

    assert path.name == "blla.mlmodel"
    assert path.is_file()


# --- Path safety ---
# Tests weights cannot escape INFERENCE_ROOT. Does not test HTTP download caching.


def test_weights_source_rejects_paths_outside_ml_root():
    with pytest.raises(ValueError, match="relative to INFERENCE_ROOT"):
        resolve_weights_source("file:///etc/passwd")

    with pytest.raises(ValueError, match="within INFERENCE_ROOT"):
        resolve_weights_source("file://../pyproject.toml")


# --- Cache directory layout ---
# Tests on-disk weight and cache folder conventions. Does not populate the cache.


def test_bundled_weights_dir_resolves_from_registry_source():
    registry = load_registry()
    uri = registry.models["syriac-calamariv1"].versions["stable"].weights_source

    assert (
        _bundled_weights_dir(uri)
        == DEFAULT_WEIGHTS_ROOT / "calamari" / "syriac-calamariv1"
    )


def test_weight_cache_layout():
    cache_dir = _weight_cache_dir("greek-calamariv1", "stable")
    assert cache_dir == DEFAULT_CACHE_ROOT / "greek-calamariv1" / "stable"
    assert DEFAULT_WEIGHTS_ROOT.name == "weights"
    assert (DEFAULT_WEIGHTS_ROOT / "calamari" / "syriac-calamariv1").is_dir()
    assert (DEFAULT_WEIGHTS_ROOT / "calamari" / "greek-calamariv1").is_dir()
    assert (DEFAULT_WEIGHTS_ROOT / "kraken" / "kraken-blla").is_dir()
