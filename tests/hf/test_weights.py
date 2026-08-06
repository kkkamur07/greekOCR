"""Essential local bundled weight resolution."""

from inference.registry import load_registry
from inference.weights import local_bundled_root, resolve_weights_source


def test_local_bundled_weights_resolve_file_uri():
    uri = "file://local/syriac/calamari/v1/stable/best.pt"
    path = resolve_weights_source(uri)

    assert path == local_bundled_root() / "local/syriac/calamari/v1/stable/best.pt"
    assert path.is_file()


def test_registry_syriac_weights_source_is_hf():
    uri = load_registry().models["syriac-calamari-v1"].versions["stable"].weights_source

    assert uri == "hf://kkkamur07/syriac-htr-calamari@stable"
