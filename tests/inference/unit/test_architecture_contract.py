"""One contract, exercised against every architecture execution path.

Calamari and BLLA are structurally parallel - resolve an artifact, preprocess,
run a session, decode, map errors - but until now each had only bespoke tests,
so a rule one path honoured and the other did not could drift indefinitely.
These tests are written against the two shared seams rather than against either
architecture, and every case runs over both:

* ``architectures.artifact.resolve_artifact`` - the order of the preflight
  checks, and therefore the HTTP status a broken deployment answers with.
* ``architectures.isolation.reraise_if_none_survived`` - partial pages survive,
  an entirely failed page re-raises its first cause with the original type.

The HTTP assertions go through ``run_errors.http_exception_for_run_error``
deliberately. It is the only thing that reads these exception types, and a
refactor that collapsed two of them into one would be invisible to a test that
only asserted the type.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inference.architectures.blla.blla_decoder import DecodedBLLALine
from inference.architectures.calamari import adapter as calamari_adapter
from inference.architectures.calamari.onnx import TranscribeLineFailure
from inference.contracts.transcribe import CharacterConfidence, TranscribeRunResponse
from inference.run_errors import http_exception_for_run_error

REPO_ROOT = Path(__file__).resolve().parents[3]


# --- Artifact preflight contract ---------------------------------------------
# Every execution path must reject a missing, foreign, or corrupt artifact in
# the same order and with types that carry the same HTTP status.


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


def _run_blla_onnx(artifact: Path, artifact_sha256: str | None) -> object:
    from inference.architectures.blla.onnx import run_blla_onnx_segment

    return run_blla_onnx_segment(
        _page_bytes(),
        model_path=artifact,
        artifact_sha256=artifact_sha256,
    )


def _execution_paths() -> list[ExecutionPath]:
    from inference.architectures.blla.blla import BLLAUnavailableError
    from inference.architectures.blla.onnx import BLLAOnnxUnavailableError

    return [
        # Calamari accepts both suffixes through one preflight, so the artifact
        # it must refuse has to sit outside the pair.
        ExecutionPath(
            name="calamari-onnx",
            native_suffix=".onnx",
            foreign_suffix=".safetensors",
            unusable_error=calamari_adapter.CalamariUnavailableError,
            run=_run_calamari,
        ),
        ExecutionPath(
            name="calamari-torch",
            native_suffix=".pt",
            foreign_suffix=".safetensors",
            unusable_error=calamari_adapter.CalamariUnavailableError,
            run=_run_calamari,
        ),
        ExecutionPath(
            name="blla-native",
            native_suffix=".safetensors",
            foreign_suffix=".onnx",
            unusable_error=BLLAUnavailableError,
            run=_run_blla_native,
        ),
        ExecutionPath(
            name="blla-onnx",
            native_suffix=".onnx",
            foreign_suffix=".safetensors",
            unusable_error=BLLAOnnxUnavailableError,
            run=_run_blla_onnx,
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

    # 503, never 404: the registry entry resolved, the file behind it did not.
    with pytest.raises(FileNotFoundError) as caught:
        path.run(absent, None)
    assert http_exception_for_run_error(caught.value).status_code == 503


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
    assert http_exception_for_run_error(caught.value).status_code == 503


@pytest.mark.parametrize("path", EXECUTION_PATHS, ids=PATH_IDS)
def test_corrupt_artifact_is_a_service_error_not_a_client_error(
    path: ExecutionPath,
    tmp_path: Path,
) -> None:
    """``ArtifactIntegrityError`` subclasses ``ValueError`` and must stay a 503."""
    from src.hf.resolve.artifacts import ArtifactIntegrityError

    artifact = tmp_path / f"model{path.native_suffix}"
    artifact.write_bytes(b"content that does not match the pinned digest")

    with pytest.raises(ArtifactIntegrityError) as caught:
        path.run(artifact, "0" * 64)

    assert isinstance(caught.value, ValueError)
    assert http_exception_for_run_error(caught.value).status_code == 503


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


# --- Per-line isolation contract ---------------------------------------------
# Both architectures split a page into units and must reach the same verdict:
# any survivor makes the page a partial success, no survivor re-raises the
# first cause with its type intact.


def _transcribed(text: str) -> TranscribeRunResponse:
    return TranscribeRunResponse(
        text=text,
        confidence=0.9,
        character_confidences=[CharacterConfidence(char=char, confidence=0.9) for char in text],
    )


def _calamari_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    unit_failures: list[Exception | None],
) -> object:
    """Run a Calamari page where the listed units fail."""
    checkpoint = tmp_path / "calamari.onnx"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setattr(
        calamari_adapter,
        "run_calamari_onnx_transcribe_many",
        lambda line_images, **_kwargs: [
            TranscribeLineFailure(index=index, error=failure)
            if failure is not None
            else _transcribed("a")
            for index, failure in enumerate(unit_failures)
        ],
    )
    return calamari_adapter.run_calamari_transcribe_many(
        [_page_bytes() for _ in unit_failures],
        checkpoint_path=checkpoint,
    )


def _blla_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    unit_failures: list[Exception | None],
) -> object:
    """Run a BLLA page where the listed decoded lines fail refinement."""
    from inference.architectures.blla import blla_runtime

    ceiling = [[10.0, 10.0], [90.0, 10.0], [90.0, 30.0], [10.0, 30.0]]
    decoded = [
        DecodedBLLALine(baseline=[[15.0, 20.0], [85.0, 20.0]], polygon=ceiling)
        for _ in unit_failures
    ]
    monkeypatch.setattr(
        "inference.architectures.blla.blla_runtime.decode_blla_heatmaps",
        lambda *_args, **_kwargs: decoded,
    )

    calls = {"index": 0}

    def refine(_image, contour, **_kwargs):
        failure = unit_failures[calls["index"]]
        calls["index"] += 1
        if failure is not None:
            raise failure
        from inference.preprocessing.segment_refinement import SegmentRefinementResult

        return [
            SegmentRefinementResult(
                points=contour,
                baseline=[[15.0, 20.0], [85.0, 20.0]],
                metadata={},
            )
        ]

    monkeypatch.setattr(
        "inference.architectures.blla.blla_runtime.refine_segment_candidates", refine
    )

    image = Image.new("RGB", (100, 40), "white")
    logits = np.zeros((4, 10, 25), dtype=np.float32)
    prepared = type("_Prepared", (), {"scaled_gray": np.zeros((10, 25), dtype=np.float32)})()
    return blla_runtime.build_blla_segment_response(
        image,
        logits,
        prepared,
        params={"use_otsu_refinement": True},
    )


PAGE_RUNNERS = [
    pytest.param(_calamari_page, id="calamari"),
    pytest.param(_blla_page, id="blla"),
]


def _survivors(result: object) -> int:
    """Count the units a page actually produced, whatever its response shape."""
    from inference.contracts.segment import SegmentRunResponse

    if isinstance(result, SegmentRunResponse):
        return len(result.lines)
    assert isinstance(result, list)
    return sum(not isinstance(entry, TranscribeLineFailure) for entry in result)


@pytest.mark.parametrize("run_page", PAGE_RUNNERS)
def test_one_failed_unit_does_not_discard_the_page(
    run_page,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A single bad line on a three-line page costs that line only."""
    result = run_page(monkeypatch, tmp_path, [None, ValueError("one bad unit"), None])

    assert _survivors(result) == 2


@pytest.mark.parametrize("run_page", PAGE_RUNNERS)
def test_a_page_with_no_failures_returns_every_unit(
    run_page,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = run_page(monkeypatch, tmp_path, [None, None])

    assert _survivors(result) == 2


@pytest.mark.parametrize("run_page", PAGE_RUNNERS)
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        # A broken artifact must stay a 503 and a bad request a 422. Flattening
        # the isolated failures into one error type - or into an empty page -
        # would silently merge these two answers.
        (calamari_adapter.CalamariUnavailableError("runtime is gone"), 503),
        (ValueError("caller sent nonsense geometry"), 422),
    ],
    ids=["unusable-runtime", "bad-request"],
)
def test_a_page_where_every_unit_failed_reraises_the_first_cause(
    run_page,
    failure: Exception,
    expected_status: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    later = RuntimeError("a later, less informative failure")

    with pytest.raises(type(failure)) as caught:
        run_page(monkeypatch, tmp_path, [failure, later])

    # The *first* failure, not the last, and not a generic wrapper.
    assert caught.value is failure
    assert http_exception_for_run_error(caught.value).status_code == expected_status


def test_an_empty_page_is_where_the_two_architectures_legitimately_differ(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A page with no units at all is not the isolation policy's business.

    The shared rule only covers a page that *failed* and produced nothing. A
    page with zero units is architecture-specific and must stay that way,
    because the units do not mean the same thing: Calamari's are line regions
    the caller supplied, so asking for none of them is a malformed request
    (422), while BLLA's are lines the decoder discovered, so finding none is a
    blank page and a legitimate empty response. Neither may re-raise, and the
    contract test pins that they do not converge by accident.
    """
    from inference.contracts.segment import SegmentRunResponse

    with pytest.raises(ValueError) as caught:
        _calamari_page(monkeypatch, tmp_path, [])
    assert http_exception_for_run_error(caught.value).status_code == 422

    blla_result = _blla_page(monkeypatch, tmp_path, [])
    assert isinstance(blla_result, SegmentRunResponse)
    assert blla_result.lines == []
    assert blla_result.blocks == []


# --- No-Torch boundary --------------------------------------------------------


def test_the_shared_seam_does_not_pull_torch_into_the_onnx_only_paths() -> None:
    """The frozen helper ships these modules and must stay Torch-free.

    Checked against the real import graph in a fresh interpreter rather than by
    inspection: the seam modules sit between the ONNX adapters and the runner,
    so a stray top-level import in either of them would add Torch to a bundle
    built to exclude it, and the bundle verifier only catches that at release.
    """
    program = (
        "import importlib, sys\n"
        "for name in (\n"
        "    'inference.architectures.artifact',\n"
        "    'inference.architectures.isolation',\n"
        "    'inference.architectures.calamari',\n"
        "    'inference.architectures.calamari.adapter',\n"
        "    'inference.architectures.calamari.onnx',\n"
        "    'inference.architectures.blla',\n"
        "    'inference.architectures.blla.onnx',\n"
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
        f"Torch leaked into the ONNX-only import graph: {completed.stdout}"
    )
