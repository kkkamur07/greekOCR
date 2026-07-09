"""registry.yaml validation and inference-local weight path helpers."""

import pytest
from inference.contracts import ComputeDevice, InferenceTask, RegistryArchitecture
from inference.registry import get_model_entry, load_registry
from inference.weights import DEFAULT_WEIGHTS_ROOT, resolve_weights_source


# --- registry.yaml entries ---
# Tests bundled model metadata loads correctly. Does not run inference.


def test_registry_yaml_validates_model_entries():
    registry = load_registry()

    syriac = get_model_entry(registry, "syriac-calamari-v1", "stable")
    assert syriac.task == InferenceTask.transcribe
    assert syriac.architecture == RegistryArchitecture.calamari
    assert syriac.device == ComputeDevice.cpu
    assert syriac.versions["stable"].weights_source.startswith("hf://")

    calamari = get_model_entry(registry, "greek-calamari-v1", "stable")
    assert calamari.task == InferenceTask.transcribe
    assert calamari.architecture == RegistryArchitecture.calamari
    assert calamari.device == ComputeDevice.cpu
    assert calamari.versions["stable"].weights_source.startswith("hf://")

    kraken = get_model_entry(registry, "greek-kraken-segment-v1", "stable")
    assert kraken.task == InferenceTask.segment
    assert kraken.architecture == RegistryArchitecture.kraken_segment
    assert kraken.device == ComputeDevice.cpu
    assert kraken.versions["stable"].weights_source == "package://kraken/blla.mlmodel"


# --- Weight path resolution ---
# Tests package:// URIs resolve to real paths. Hub local/hf:// paths live in tests/hf.


def test_registry_package_weights_source_resolves_package_resource():
    registry = load_registry()
    uri = registry.models["greek-kraken-segment-v1"].versions["stable"].weights_source
    path = resolve_weights_source(uri)

    assert path.name == "blla.mlmodel"
    assert path.is_file()


# --- Path safety ---
# Tests weights cannot escape INFERENCE_ROOT. Does not test Hub download caching.


def test_weights_source_rejects_paths_outside_ml_root():
    with pytest.raises(ValueError, match="relative to INFERENCE_ROOT"):
        resolve_weights_source("file:///etc/passwd")

    with pytest.raises(ValueError, match="within INFERENCE_ROOT"):
        resolve_weights_source("file://../pyproject.toml")


def test_interim_weights_layout():
    assert DEFAULT_WEIGHTS_ROOT.name == "weights"
    assert (DEFAULT_WEIGHTS_ROOT / "kraken").is_dir()
