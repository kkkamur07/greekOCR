"""registry.yaml validation and weight path helpers."""

from ml.contracts import ComputeDevice, MLTask, RegistryArchitecture
from ml.registry import DEFAULT_REGISTRY_PATH, get_model_entry, load_registry
from ml.weights import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_WEIGHTS_ROOT,
    resolve_weights_source,
    weight_cache_dir,
)


def test_registry_yaml_loads_transcribe_and_segment_models():
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


def test_registry_path_points_at_repo_file():
    assert DEFAULT_REGISTRY_PATH.is_file()


def test_weights_source_resolves_under_ml_root():
    registry = load_registry()
    uri = registry.models["greek-calamariv1"].versions["stable"].weights_source
    path = resolve_weights_source(uri)

    assert path == (DEFAULT_REGISTRY_PATH.parent / "weights/calamari/greek-calamariv1/stable.ckpt")


def test_weight_cache_layout():
    cache_dir = weight_cache_dir("greek-calamariv1", "stable")
    assert cache_dir == DEFAULT_CACHE_ROOT / "greek-calamariv1" / "stable"
    assert DEFAULT_WEIGHTS_ROOT.name == "weights"
    assert (DEFAULT_WEIGHTS_ROOT / "calamari" / "greek-calamariv1").is_dir()
    assert (DEFAULT_WEIGHTS_ROOT / "kraken" / "kraken-blla").is_dir()
