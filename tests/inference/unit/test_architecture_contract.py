"""One contract, exercised against every architecture execution path.

Calamari and BLLA are structurally parallel - resolve an artifact, preprocess,
run a session, decode, map errors - but until now each had only bespoke tests,
so a rule one path honoured and the other did not could drift indefinitely.
These tests are written against the shared seam rather than against either
architecture, and every case runs over both:

* ``architectures.artifact.resolve_artifact`` - the order of the preflight
  checks, and therefore which of the two failure families a broken deployment
  lands in.

The per-line isolation half of this file was removed: it stubbed the decoder
*and* the refiner for BLLA and the whole ONNX batch seam for Calamari, leaving
a single production ``for`` loop executing. Every claim it made is made against
real code in ``test_transcribe_batch_isolation``,
``test_blla_polygonization_isolation`` and ``test_segment_refinement``.

The two families are asserted by type, not by HTTP status. Until #60 they were
read by ``run_errors.http_exception_for_run_error``, which turned a
``RuntimeError`` into a 503 and a ``ValueError`` into a 422 for the loopback
service; that service is gone and an **inference agent** reports a failed page
in words through the job callback. The distinction still has to hold, because
``ArtifactIntegrityError`` subclasses ``ValueError`` on purpose and is a broken
deployment rather than a bad request - it just no longer has a status attached.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from inference.architectures.calamari import adapter as calamari_adapter

REPO_ROOT = Path(__file__).resolve().parents[3]


# --- Artifact preflight contract ---------------------------------------------
# Every execution path must reject a missing, foreign, or corrupt artifact in
# the same order and with types from the same failure family.


@dataclass(frozen=True)
class ExecutionPath:
    """One architecture's real entry point, reduced to what the contract needs."""

    name: str
    native_suffix: str
    foreign_suffix: str
    unusable_error: type[RuntimeError]
    run: Callable[[Path, str | None], object]


def _page_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), "white").save(output, format="PNG")
    return output.getvalue()


def _run_calamari(artifact: Path, artifact_sha256: str | None) -> object:
    return calamari_adapter.run_calamari_transcribe_many(
        [_page_bytes()],
        checkpoint_path=artifact,
        artifact_sha256=artifact_sha256,
    )


def _run_blla_native(artifact: Path, artifact_sha256: str | None) -> object:
    from inference.architectures.blla.blla import run_blla_segment

    return run_blla_segment(
        _page_bytes(),
        model_path=artifact,
        artifact_sha256=artifact_sha256,
    )


def _execution_paths() -> list[ExecutionPath]:
    from inference.architectures.blla.blla import BLLAUnavailableError

    # One execution path per architecture, and under ADR 0006 both load the
    # same format. The foreign suffix is each architecture's *retired* native
    # format, which is the confusion that actually happens: those files are
    # published at the same Hub revision and land in the same cache directory,
    # so a preflight that shrugged at them would run the wrong file.
    return [
        ExecutionPath(
            name="calamari-onnx",
            native_suffix=".onnx",
            foreign_suffix=".pt",
            unusable_error=calamari_adapter.CalamariUnavailableError,
            run=_run_calamari,
        ),
        ExecutionPath(
            name="blla-onnx",
            native_suffix=".onnx",
            foreign_suffix=".safetensors",
            unusable_error=BLLAUnavailableError,
            run=_run_blla_native,
        ),
    ]


EXECUTION_PATHS = _execution_paths()
PATH_IDS = [path.name for path in EXECUTION_PATHS]


@pytest.mark.parametrize("path", EXECUTION_PATHS, ids=PATH_IDS)
def test_missing_artifact_is_a_service_error_on_every_path(
    path: ExecutionPath,
    tmp_path: Path,
) -> None:
    absent = tmp_path / f"absent{path.native_suffix}"

    with pytest.raises(FileNotFoundError):
        path.run(absent, None)

    # A deployment failure, never a "no such model": the registry entry
    # resolved, the file behind it did not.
    with pytest.raises(FileNotFoundError) as caught:
        path.run(absent, None)
    assert not isinstance(caught.value, ValueError)


@pytest.mark.parametrize("path", EXECUTION_PATHS, ids=PATH_IDS)
def test_foreign_artifact_is_a_service_error_on_every_path(
    path: ExecutionPath,
    tmp_path: Path,
) -> None:
    foreign = tmp_path / f"model{path.foreign_suffix}"
    foreign.write_bytes(b"not a model this runtime can load")

    with pytest.raises(path.unusable_error) as caught:
        path.run(foreign, None)

    # Every architecture's "unusable artifact" type must remain a RuntimeError
    # subclass. Demote one to ValueError and this deployment failure silently
    # starts telling callers their request was malformed.
    assert isinstance(caught.value, RuntimeError)


@pytest.mark.parametrize("path", EXECUTION_PATHS, ids=PATH_IDS)
def test_corrupt_artifact_is_a_service_error_not_a_client_error(
    path: ExecutionPath,
    tmp_path: Path,
) -> None:
    """``ArtifactIntegrityError`` subclasses ``ValueError`` but is not a bad request."""
    from inference.hub.artifacts import ArtifactIntegrityError

    artifact = tmp_path / f"model{path.native_suffix}"
    artifact.write_bytes(b"content that does not match the pinned digest")

    with pytest.raises(ArtifactIntegrityError) as caught:
        path.run(artifact, "0" * 64)

    assert isinstance(caught.value, ValueError)
    assert type(caught.value) is ArtifactIntegrityError


@pytest.mark.parametrize("path", EXECUTION_PATHS, ids=PATH_IDS)
def test_preflight_reports_the_first_problem_in_a_fixed_order(
    path: ExecutionPath,
    tmp_path: Path,
) -> None:
    """Existence beats suffix beats digest, identically on every path.

    The order is not cosmetic: hashing a file that is not there raises ``OSError``
    from the hasher instead of the ``FileNotFoundError`` the mapping expects.
    """
    absent_and_foreign = tmp_path / f"absent{path.foreign_suffix}"
    with pytest.raises(FileNotFoundError):
        path.run(absent_and_foreign, "0" * 64)

    foreign_and_corrupt = tmp_path / f"present{path.foreign_suffix}"
    foreign_and_corrupt.write_bytes(b"wrong format and wrong digest")
    with pytest.raises(path.unusable_error):
        path.run(foreign_and_corrupt, "0" * 64)


# --- Runtime boundary ---------------------------------------------------------


def test_no_torch_remains_in_the_inference_import_graph() -> None:
    """ADR 0006 retired PyTorch from the runtime; nothing may drag it back in.

    This is the guard that keeps the published closure honest. Torch is 475 MB
    of the 817 MB a researcher used to install, and it is *still in this
    repository* - `src/model/inference_export/` traces the graph with it - so the only thing
    standing between the two is that nothing under `inference/` imports it. A
    single convenience import would put it back in `[project].dependencies`
    without anyone noticing, because the dev venv has Torch installed and
    everything would keep working here.

    Checked against the real import graph in a fresh interpreter rather than by
    reading source, so a re-export through some third module is caught too. The
    inverse of this test guarded ADR 0004; the pair of them have now traded
    places twice, which is the argument for asserting on the import graph
    instead of on a denylist somebody maintains.
    """
    program = (
        "import importlib, sys\n"
        "for name in (\n"
        "    'inference.architectures.artifact',\n"
        "    'inference.architectures.isolation',\n"
        "    'inference.architectures.calamari',\n"
        "    'inference.architectures.calamari.adapter',\n"
        "    'inference.architectures.blla',\n"
        "    'inference.architectures.blla.blla',\n"
        "    'inference.architectures.blla.blla_runtime',\n"
        "    'inference.jobs.runner',\n"
        "):\n"
        "    importlib.import_module(name)\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in {'torch', 'torchvision', 'safetensors', 'kraken'}\n"
        ")\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        f"a retired runtime is back in the inference import graph: {completed.stdout}"
    )


# `test_both_architectures_run_on_cpu_only` stood here and asserted `"cuda" not in
# source` and `"mps" not in source` over four hand-listed files. It was a strict subset
# of `test_the_runtime_never_selects_an_accelerator` in test_torch_runtime.py, which
# loads both real checkpoints and asserts every parameter's `device.type == "cpu"` --
# catching a device migration however it is spelled, and in any module, not just the
# four named here. The grep additionally went red on the word "cuda" in a comment and
# was blind to `calamari/model.py` and `calamari/layers.py`. That test carries no `ml`
# marker, so it runs in the same pull-request lane this one did: no coverage was lost.
