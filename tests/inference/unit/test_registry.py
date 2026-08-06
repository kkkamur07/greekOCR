"""registry.yaml validation and inference-local weight path helpers."""

import pytest

from inference.contracts import ComputeDevice, InferenceTask, RegistryArchitecture
from inference.registry import RegistryVersionEntry, get_model_entry, load_registry
from inference.weights import resolve_weights_source

# --- registry.yaml entries ---
# Tests bundled model metadata loads correctly. Does not run inference.


def test_registry_yaml_validates_model_entries():
    """The schema each entry must satisfy, not the values one publish happened to have.

    The literal ``hub_revision`` and ``artifact_sha256`` this used to inline made a
    legitimate model republish a test edit. What the digests are actually *worth* is
    checked against real bytes by ``test_onnx_runtime.py::
    test_the_registry_pins_the_digest_of_the_artifact_the_loader_opens``; here only
    their presence and shape matter.
    """
    registry = load_registry()

    syriac = get_model_entry(registry, "syriac-calamari-v1", "stable")
    assert syriac.task == InferenceTask.transcribe
    assert syriac.architecture == RegistryArchitecture.calamari
    assert syriac.device == ComputeDevice.cpu
    assert syriac.host_eligibility.value == "local"
    assert syriac.versions["stable"].weights_source.startswith("hf://")
    assert len(syriac.versions["stable"].hub_revision) == 40
    assert len(syriac.versions["stable"].artifact_sha256) == 64

    blla = get_model_entry(registry, "blla-segment", "stable")
    assert blla.task == InferenceTask.segment
    assert blla.architecture == RegistryArchitecture.blla
    assert blla.device == ComputeDevice.cpu
    assert blla.versions["stable"].weights_source.startswith("hf://")
    assert len(blla.versions["stable"].hub_revision) == 40
    assert len(blla.versions["stable"].artifact_sha256) == 64


# --- Weight path resolution ---
# Hub weights are resolved in tests/hf; this verifies the native artifact shape.


def test_registry_rejects_partial_hf_provenance():
    with pytest.raises(ValueError, match="both hub_revision and artifact_sha256"):
        RegistryVersionEntry(
            weights_source="hf://example/demo@stable",
            hub_revision="a" * 40,
        )


# `test_blla_rejects_digest_mismatch_before_runtime_load` stood here. The same ordering
# claim -- digest before open, integrity failure rather than a parse error -- is
# `test_architecture_contract.py::test_corrupt_artifact_is_a_service_error_not_a_client_error`,
# which runs it over both architectures instead of BLLA alone.


# --- Path safety ---
# Tests weights cannot escape INFERENCE_ROOT. Does not test Hub download caching.


def test_weights_source_rejects_paths_outside_ml_root():
    with pytest.raises(ValueError, match="relative to INFERENCE_ROOT"):
        resolve_weights_source("file:///etc/passwd")

    with pytest.raises(ValueError, match="within INFERENCE_ROOT"):
        resolve_weights_source("file://../pyproject.toml")


# `test_interim_weights_layout` stood here and asserted `inference/weights/kraken/` is a
# directory. pyproject.toml calls that tree an empty placeholder from the pre-Hub weights
# layout and excludes its `.gitkeep` from the wheel; `DEFAULT_WEIGHTS_ROOT` has no
# production caller.
