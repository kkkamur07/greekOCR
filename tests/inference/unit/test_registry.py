"""registry.yaml validation and inference-local weight path helpers."""

import pytest
from inference.contracts import ComputeDevice, InferenceTask, RegistryArchitecture
from inference.registry import RegistryVersionEntry, get_model_entry, load_registry
from inference.weights import DEFAULT_WEIGHTS_ROOT, resolve_weights_source

# --- registry.yaml entries ---
# Tests bundled model metadata loads correctly. Does not run inference.


def test_registry_yaml_validates_model_entries():
    registry = load_registry()

    syriac = get_model_entry(registry, "syriac-calamari-v1", "stable")
    assert syriac.task == InferenceTask.transcribe
    assert syriac.architecture == RegistryArchitecture.calamari
    assert syriac.device == ComputeDevice.cpu
    assert syriac.host_eligibility.value == "local"
    assert syriac.versions["stable"].weights_source.startswith("hf://")
    assert syriac.versions["stable"].hub_revision == "5ff715e873f1ae3f325ebea4d2c4a95eb5094601"
    assert (
        syriac.versions["stable"].artifact_sha256
        == "ea711b918010aa31bd4a8a5de99c7953207421a7c7d4a39163166db380013053"
    )
    assert "greek-calamari-v1" not in registry.models

    blla = get_model_entry(registry, "blla-segment", "stable")
    assert blla.task == InferenceTask.segment
    assert blla.architecture == RegistryArchitecture.blla
    assert blla.device == ComputeDevice.cpu
    assert blla.versions["stable"].weights_source == "hf://kkkamur07/segmentation-blla@stable"
    assert (
        blla.versions["stable"].artifact_sha256
        == "8b5b6ec21947d3c5a7c117c7b873f3a923aab48950ba05707805899877c397ce"
    )
    assert blla.versions["stable"].hub_revision == "444d51dd7b34cd2012b1ffe1c9c9442c875d8230"


# --- Weight path resolution ---
# Hub weights are resolved in tests/hf; this verifies the native artifact shape.


def test_registry_rejects_partial_hf_provenance():
    with pytest.raises(ValueError, match="both hub_revision and artifact_sha256"):
        RegistryVersionEntry(
            weights_source="hf://example/demo@stable",
            hub_revision="a" * 40,
        )


def test_native_blla_rejects_digest_mismatch_before_runtime_load():
    from inference.architectures.blla.blla import run_blla_segment

    model_path = (
        DEFAULT_WEIGHTS_ROOT.parent.parent
        / "src"
        / "hf"
        / "staging"
        / "models"
        / "segmentation"
        / "blla"
        / "v1"
        / "stable"
        / "blla.safetensors"
    )
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        run_blla_segment(
            b"not-read-after-integrity-failure",
            model_path=model_path,
            artifact_sha256="0" * 64,
        )


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
