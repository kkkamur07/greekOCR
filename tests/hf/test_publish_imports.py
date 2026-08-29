"""The publish scripts import `src.hf.publish` lazily, inside `main()`.

Nothing else in the suite imports that package, so the rename of the
`inference` import package to `nomikos_inference` broke
`src.hf.publish.staging` and every check stayed green: a stale module path
inside a function body is invisible until someone runs the script. These tests
import the package eagerly, and name the symbols each script actually pulls
out of it, so the next rename fails here instead of at the command line.
"""

from __future__ import annotations

import importlib

import pytest

# The names each scripts/hf entry point imports inside its own `main()`.
SCRIPT_IMPORTS = {
    "publish_model": (
        "build_model_card",
        "get_default_publish_client",
        "plan_model_publish",
        "publish_model",
        "upload_enabled",
    ),
    "publish_dataset": (
        "build_dataset_readme",
        "plan_dataset_publish",
        "publish_dataset",
        "upload_enabled",
    ),
    "sync_collection": (
        "load_collection_spec",
        "plan_collection_sync",
        "sync_collection",
    ),
}


def test_publish_package_imports():
    """`src.hf.publish.staging` reaches its `nomikos_inference` dependency."""
    importlib.import_module("src.hf.publish")


@pytest.mark.parametrize("script", sorted(SCRIPT_IMPORTS))
def test_script_deferred_imports_resolve(script: str):
    """Every name a scripts/hf entry point imports at call time exists."""
    module = importlib.import_module("src.hf.publish")
    missing = [name for name in SCRIPT_IMPORTS[script] if not hasattr(module, name)]
    assert not missing, f"scripts/hf/{script}.py imports missing names: {missing}"
