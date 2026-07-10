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
    assert syriac.versions["stable"].hub_revision == "e01349626f934ee9526d486a42b5c4173b8d7a26"
    assert (
        syriac.versions["stable"].artifact_sha256
        == "ea711b918010aa31bd4a8a5de99c7953207421a7c7d4a39163166db380013053"
    )

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
    assert (
        kraken.versions["stable"].artifact_sha256
        == "77a638a83c9e535620827a09e410ed36391e9e8e8126d5796a0f15b978186056"
    )


# --- Weight path resolution ---
# Tests package:// URIs resolve to real paths. Hub local/hf:// paths live in tests/hf.


def test_registry_package_weights_source_resolves_package_resource():
    registry = load_registry()
    version = registry.models["greek-kraken-segment-v1"].versions["stable"]
    path = resolve_weights_source(
        version.weights_source,
        artifact_sha256=version.artifact_sha256,
    )

    assert path.name == "blla.mlmodel"
    assert path.is_file()


def test_registry_rejects_partial_hf_provenance():
    with pytest.raises(ValueError, match="both hub_revision and artifact_sha256"):
        RegistryVersionEntry(
            weights_source="hf://example/demo@stable",
            hub_revision="a" * 40,
        )


def test_package_weights_reject_digest_mismatch():
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        resolve_weights_source(
            "package://kraken/blla.mlmodel",
            artifact_sha256="0" * 64,
        )


def test_kraken_rejects_digest_mismatch_before_runtime_load():
    from inference.architectures.kraken import run_kraken_segment

    model_path = resolve_weights_source("package://kraken/blla.mlmodel")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        run_kraken_segment(
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
