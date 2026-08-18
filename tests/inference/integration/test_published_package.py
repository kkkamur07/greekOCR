"""The published `nomikos-inference` package, exercised as an installed wheel.

Everything here runs against a real wheel installed into a real, empty virtual
environment, in a subprocess whose working directory is outside the repository.
That last detail is the whole point: run this from the repository root with the
tree importable and every assertion would pass whether or not the package
boundary exists. The subprocess proves it does.

Marked `ml` because it downloads a dependency closure and real **Hub
artifact**s, and runs both architectures on a real page - minutes, not seconds.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.paths import REPO_ROOT, SEGMENT_PAGE, TRANSCRIBE_LINE

pytestmark = pytest.mark.ml

# ADR 0006 needs no install flag at all. Under ADR 0004 this constant was
# `--torch-backend=cpu`, and it was not a nicety: without it a Linux resolve
# pulled sixteen nvidia/triton wheels behind `torch` and this fixture alone
# installed about 4.8 GB. `onnxruntime` publishes one CPU wheel per platform,
# so there is no accelerator variant to exclude and nothing for a researcher to
# remember.

# Every platform the package claims to support. Intel macOS is back: it was
# absent under ADR 0004 only because PyTorch publishes no
# `x86_64-apple-darwin` wheel from 2.10 onward.
TARGET_PLATFORMS = (
    "x86_64-manylinux_2_28",
    "aarch64-manylinux_2_28",
    "x86_64-pc-windows-msvc",
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
)


def _uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        pytest.skip("uv is required to build and install the published package")
    return executable


@pytest.fixture(scope="session")
def installed_package(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the wheel and install it into an otherwise empty environment.

    Session-scoped: this costs a dependency-closure download, and every test in
    the module is asking a question about the same installed artifact.
    """
    uv = _uv()
    workspace = tmp_path_factory.mktemp("published-package")
    dist = workspace / "dist"
    venv = workspace / "venv"

    subprocess.run(
        [uv, "build", "--wheel", "-o", str(dist)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = sorted(dist.glob("nomikos_inference-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    subprocess.run(
        [uv, "venv", str(venv), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"],
        check=True,
        capture_output=True,
        text=True,
    )
    python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(wheels[0])],
        check=True,
        capture_output=True,
        text=True,
    )

    return {"wheel": wheels[0], "venv": venv, "python": python, "elsewhere": workspace}


def _run_installed(
    installed_package: dict[str, Path],
    source: str,
    *,
    env: dict[str, str] | None = None,
) -> str:
    """Run `source` under the installed interpreter, outside the repository.

    `cwd` is the throwaway workspace and `PYTHONPATH` is cleared, so an import
    of `inference` can only be satisfied by the installed wheel.

    The repository-relative settings the test session runs under are dropped
    too. `INFERENCE_REGISTRY_PATH=inference/registry.yaml` is the load-bearing
    one: leaving it set would have the installed package read the **Registry**
    out of the checkout, and finding its own bundled copy is part of what is
    being tested.
    """
    environment = dict(os.environ)
    for leaked in ("PYTHONPATH", "INFERENCE_REGISTRY_PATH", "HF_CACHE_ROOT"):
        environment.pop(leaked, None)
    environment.update(env or {})
    completed = subprocess.run(
        [str(installed_package["python"]), "-c", source],
        cwd=installed_package["elsewhere"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _last_json_line(output: str) -> dict:
    return json.loads(output.strip().splitlines()[-1])


# Five tests stood here, and between them they tested pip, hatchling and CPython's import
# system rather than this repository's code:
#
# * `..._imports_from_site_packages_not_the_repository_tree` asserted that clearing
#   `PYTHONPATH` makes an import resolve out of site-packages. That is the stdlib's
#   behaviour, and `_run_installed` arranges it for every test below anyway.
# * `test_the_console_entry_point_is_present_and_executable` asserted `[project.scripts]`
#   produced a file. `test_cli_pairing.py` and `test_cli_run.py` both assert the console
#   script exists in their `installed_cli` fixtures and then *run* it, and
#   `test_cli_pairing.py::test_the_version_subcommand_reports_the_installed_package_version`
#   keeps the `0+unknown` guard in the lane that runs on every pull request rather than
#   behind this module's `ml` marker.
# * `test_the_installed_closure_carries_no_torch_and_no_accelerator_wheels` was a strict
#   subset of `test_every_target_platform_resolves_without_an_accelerator_wheel` below,
#   which asserts the same three properties across five target platforms instead of one.
# * `test_the_installed_package_holds_no_web_server` asserted `find_spec` returns `None`
#   for `inference.api` and `inference.helper`, two modules #60 deleted; asking whether a
#   module that does not exist can be found is not a guard. Its `fastapi`/`uvicorn`/
#   `starlette` half is a closure claim, also covered five platforms wide below.
# * `test_the_installed_package_opens_no_socket` monkeypatched `socket.socket.bind` and
#   imported four modules. No library binds a socket at import time, so it could only ever
#   pass.


def test_the_hub_cache_defaults_under_the_researchers_home_directory(
    installed_package: dict[str, Path],
) -> None:
    """Not beside the code: in a wheel, "beside the code" is site-packages."""
    output = _run_installed(
        installed_package,
        "import json, pathlib;"
        " from inference.hub.cache import DEFAULT_CACHE_ROOT;"
        " print(json.dumps({'root': str(DEFAULT_CACHE_ROOT),"
        " 'home': str(pathlib.Path.home())}))",
    )
    result = _last_json_line(output)

    assert Path(result["root"]).is_relative_to(Path(result["home"]))
    assert not Path(result["root"]).is_relative_to(installed_package["venv"])


@pytest.fixture(scope="session")
def real_page_run(installed_package: dict[str, Path], tmp_path_factory) -> dict:
    """Segment and transcribe a real page through the installed package.

    One session-scoped run rather than one per assertion: it downloads both
    **Hub artifact**s and runs both architectures, and every question below is
    about the same execution.
    """
    cache_root = tmp_path_factory.mktemp("hub-cache")
    source = f"""
import json, pathlib
from inference.contracts.common import InferenceTask
from inference.jobs.runner import run_model

segment = run_model(
    task=InferenceTask.segment,
    registry_model_id="blla-segment",
    registry_tag="stable",
    image_bytes=pathlib.Path({str(SEGMENT_PAGE)!r}).read_bytes(),
)
transcribe = run_model(
    task=InferenceTask.transcribe,
    registry_model_id="syriac-calamari-v1",
    registry_tag="stable",
    image_bytes=pathlib.Path({str(TRANSCRIBE_LINE)!r}).read_bytes(),
    params={{"line_index": 0}},
)
print(json.dumps({{
    "lines": len(segment.lines),
    "blocks": len(segment.blocks),
    "adapter": segment.lines[0].source_metadata.get("adapter"),
    "text": transcribe.text,
    "confidence": transcribe.confidence,
}}))
"""
    output = _run_installed(installed_package, source, env={"HF_CACHE_ROOT": str(cache_root)})
    return {"result": _last_json_line(output), "cache_root": cache_root}


def test_a_real_page_is_segmented_and_transcribed_through_the_installed_package(
    real_page_run: dict,
) -> None:
    result = real_page_run["result"]

    assert result["blocks"] == 1
    assert result["lines"] > 1
    assert result["adapter"] == "blla"
    assert result["text"].strip() != ""
    assert 0.0 <= result["confidence"] <= 1.0


# `test_the_installed_package_resolves_hf_weights_and_records_their_provenance` stood here
# and asserted the manifest's `hub_revision` and `artifact_sha256` are 40 and 64 characters
# long. That is `tests/hf/test_resolve.py`, and the test below makes the stronger claim
# about the same two manifests: not that the fields have the right *shape* but that they
# match the registry's pins and the bytes on disk.


def test_the_cached_artifacts_match_the_digests_the_registry_pins(
    real_page_run: dict,
) -> None:
    """**Artifact SHA-256**, verified independently of the code that verified it.

    Hashing the bytes here rather than trusting the manifest is what makes this
    a check on the resolver instead of a check on its own bookkeeping.
    """
    import yaml

    registry = yaml.safe_load((REPO_ROOT / "inference" / "registry.yaml").read_text())
    cache_root = real_page_run["cache_root"]

    checked = 0
    for registry_model_id, entry in registry["models"].items():
        version = entry["versions"]["stable"]
        cache_dir = cache_root / registry_model_id / "stable"
        if not cache_dir.is_dir():
            continue
        manifest = json.loads((cache_dir / ".hub-manifest.json").read_text())
        artifact = cache_dir / manifest["artifact_path"]

        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == version["artifact_sha256"]
        assert manifest["hub_revision"] == version["hub_revision"]
        checked += 1

    assert checked == 2


# `test_a_corrupted_artifact_is_rejected_by_the_installed_verifier` stood here. Accept and
# reject for `verify_artifact_sha256` are `tests/hf/test_artifacts.py`; this ran the same
# two calls in a subprocess to show the function is in the wheel, which `real_page_run`
# above already shows by resolving two digest-pinned artifacts through it and running them.


@pytest.mark.parametrize("platform", TARGET_PLATFORMS)
def test_every_target_platform_resolves_without_an_accelerator_wheel(
    installed_package: dict[str, Path], tmp_path: Path, platform: str
) -> None:
    """The closure has to hold where the package lands, not where it was built.

    This machine's resolution proves nothing on its own: it was Linux that
    dragged CUDA under ADR 0004, and it did so with no flag passed. Resolving
    with **no flag at all** is the claim now - the requirements come from the
    built wheel's own metadata, so the check cannot drift from what ships.
    """
    uv = _uv()
    output = _run_installed(
        installed_package,
        "import json; from importlib.metadata import metadata;"
        " print(json.dumps(metadata('nomikos-inference').get_all('Requires-Dist')))",
    )
    requirements = tmp_path / f"{platform}.in"
    requirements.write_text("\n".join(json.loads(output.strip().splitlines()[-1])) + "\n")
    resolution = tmp_path / f"{platform}.txt"

    resolved = subprocess.run(
        [
            uv,
            "pip",
            "compile",
            "--quiet",
            "--python-platform",
            platform,
            "--python-version",
            "3.11",
            str(requirements),
            "-o",
            str(resolution),
        ],
        capture_output=True,
        text=True,
    )

    assert resolved.returncode == 0, resolved.stderr
    pinned = [line.split("==")[0] for line in resolution.read_text().splitlines() if "==" in line]

    assert [name for name in pinned if name.startswith(("nvidia", "triton"))] == []
    assert [name for name in pinned if name.lower() in {"torch", "torchvision"}] == []
    assert "onnxruntime" in pinned
