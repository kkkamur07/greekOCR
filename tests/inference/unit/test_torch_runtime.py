"""The PyTorch runtime contract, exercised on real weights (ADR 0004).

Everything here runs the published checkpoints for real. The rules under test
are the ones ADR 0004 makes load-bearing and that nothing else pins:

* both architectures execute through PyTorch on CPU,
* dropout is off and no autograd graph is built on any inference path,
* the **artifact SHA-256** is verified *before* a checkpoint is opened, which
  is what keeps ``torch.load`` off an unverified pickle,
* the outputs are deterministic run to run, so a local result and a cloud
  worker result are comparable.

No stubbed sessions: a test that cannot run the real graph does not belong in
this file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from inference.architectures.artifact import artifact_fingerprint
from inference.architectures.blla.blla import BLLAUnavailableError, run_blla_segment
from inference.architectures.calamari.adapter import (
    CalamariUnavailableError,
    _load_checkpoint,
    run_calamari_transcribe,
)
from inference.architectures.calamari.checkpoint import load_calamari_checkpoint
from inference.hub.artifacts import ArtifactIntegrityError
from tests.fixtures.paths import REPO_ROOT, SEGMENT_PAGE, TRANSCRIBE_LINE

CALAMARI_CHECKPOINT = REPO_ROOT / "src/hf/local/syriac/calamari/v1/stable/best.pt"
BLLA_CHECKPOINT = REPO_ROOT / "src/hf/staging/models/segmentation/blla/v1/stable/blla.safetensors"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Both architectures run, on PyTorch, on CPU -------------------------------


def test_transcribe_runs_the_torch_graph_on_real_weights() -> None:
    response = run_calamari_transcribe(
        TRANSCRIBE_LINE.read_bytes(),
        checkpoint_path=CALAMARI_CHECKPOINT,
        artifact_sha256=_digest(CALAMARI_CHECKPOINT),
    )

    assert response.text
    assert 0.0 <= response.confidence <= 1.0
    assert len(response.character_confidences) == len(response.text)


def test_segment_runs_the_torch_graph_on_real_weights() -> None:
    response = run_blla_segment(
        SEGMENT_PAGE.read_bytes(),
        model_path=BLLA_CHECKPOINT,
        artifact_sha256=_digest(BLLA_CHECKPOINT),
    )

    assert len(response.blocks) == 1
    assert len(response.lines) > 10
    assert all(len(line.points) >= 4 for line in response.lines)


def test_both_runtimes_are_deterministic_across_runs() -> None:
    """A researcher's result must be reproducible against the cloud worker.

    Anything left in training mode - dropout above all - would show up here as
    two different answers to the same page.
    """
    first = run_calamari_transcribe(
        TRANSCRIBE_LINE.read_bytes(), checkpoint_path=CALAMARI_CHECKPOINT
    )
    second = run_calamari_transcribe(
        TRANSCRIBE_LINE.read_bytes(), checkpoint_path=CALAMARI_CHECKPOINT
    )
    assert first.text == second.text
    assert first.confidence == second.confidence

    page = SEGMENT_PAGE.read_bytes()
    first_segment = run_blla_segment(page, model_path=BLLA_CHECKPOINT)
    second_segment = run_blla_segment(page, model_path=BLLA_CHECKPOINT)
    assert [line.points for line in first_segment.lines] == [
        line.points for line in second_segment.lines
    ]


# --- eval() and inference mode ------------------------------------------------


def test_loaded_models_are_in_eval_mode_with_dropout_disabled() -> None:
    """``model.eval()`` is not cosmetic: this graph has a p=0.5 dropout layer."""
    _load_checkpoint.cache_clear()
    model, _, _ = _load_checkpoint(str(CALAMARI_CHECKPOINT))

    assert not model.training
    dropouts = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    assert dropouts, "the Calamari graph is expected to contain dropout"
    assert all(not module.training for module in dropouts)

    from inference.architectures.blla.blla import _load_blla_model

    _load_blla_model.cache_clear()
    blla_model = _load_blla_model(str(BLLA_CHECKPOINT))
    assert not blla_model.training


def test_no_inference_path_builds_an_autograd_graph() -> None:
    """Without ``inference_mode``/``no_grad`` every page would retain a graph.

    Observed from inside the forward pass with grad *globally enabled*, so the
    only thing that can make the assertion hold is the adapter's own scope. The
    same graph run outside that scope is checked too, or the test would pass
    just as well against a build where autograd had been disabled globally.
    """
    observed: list[bool] = []

    def record(_module, _inputs, _output) -> None:
        observed.append(torch.is_inference_mode_enabled())

    # Same cache key the adapter uses, or the hook lands on a second instance.
    model, _, line_height = _load_checkpoint(
        str(CALAMARI_CHECKPOINT), artifact_fingerprint(CALAMARI_CHECKPOINT)
    )
    handle = model.register_forward_hook(record)
    try:
        with torch.enable_grad():
            response = run_calamari_transcribe(
                TRANSCRIBE_LINE.read_bytes(), checkpoint_path=CALAMARI_CHECKPOINT
            )
            assert response.text
            assert observed == [True]

            # The same graph outside the adapter's scope does build a graph, so
            # the assertion above is about the adapter and not about the build.
            from inference.architectures.calamari.preprocessing import (
                preprocess_line_image_bytes_to_calamari_tensor,
            )

            image = preprocess_line_image_bytes_to_calamari_tensor(
                TRANSCRIBE_LINE.read_bytes(), line_height=line_height
            )
            leaked = model(torch.from_numpy(image.astype("float32")))
            assert observed == [True, False]
            assert leaked["softmax"].grad_fn is not None
    finally:
        handle.remove()


def test_the_blla_page_forward_also_runs_in_inference_mode() -> None:
    from inference.architectures.blla.blla import _load_blla_model

    observed: list[bool] = []
    model = _load_blla_model(str(BLLA_CHECKPOINT), artifact_fingerprint(BLLA_CHECKPOINT))
    handle = model.register_forward_hook(
        lambda _m, _i, _o: observed.append(torch.is_inference_mode_enabled())
    )
    try:
        with torch.enable_grad():
            run_blla_segment(SEGMENT_PAGE.read_bytes(), model_path=BLLA_CHECKPOINT)
    finally:
        handle.remove()

    assert observed == [True]


def test_the_runtime_never_selects_an_accelerator() -> None:
    """CPU only: MPS would make a laptop result differ from the cloud worker."""
    _load_checkpoint.cache_clear()
    model, _, _ = _load_checkpoint(str(CALAMARI_CHECKPOINT))

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())

    from inference.architectures.blla.blla import _load_blla_model

    _load_blla_model.cache_clear()
    blla_model = _load_blla_model(str(BLLA_CHECKPOINT))
    assert all(parameter.device.type == "cpu" for parameter in blla_model.parameters())


# --- Checkpoint loading is not a code-execution surface -----------------------


class _UnsafeCheckpointPayload:
    """A checkpoint whose unpickling would write a file."""

    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _write_marker, (str(self.marker_path),)


def _write_marker(marker_path: str) -> None:
    Path(marker_path).write_text("unsafe checkpoint deserialized")


def test_a_digest_mismatch_stops_the_load_before_torch_sees_the_file(
    tmp_path: Path,
) -> None:
    """Verification is ordered ahead of the load, not merely present.

    A pickled payload that executes on load is the exact thing the ordering
    protects against, so the test uses one: if the digest check ran after
    ``torch.load``, the marker file would exist.
    """
    checkpoint = tmp_path / "malicious.pt"
    marker = tmp_path / "executed"
    torch.save(_UnsafeCheckpointPayload(marker), checkpoint)

    with pytest.raises(ArtifactIntegrityError):
        run_calamari_transcribe(
            TRANSCRIBE_LINE.read_bytes(),
            checkpoint_path=checkpoint,
            artifact_sha256="0" * 64,
        )

    assert not marker.exists()


def test_a_verified_but_pickled_checkpoint_is_still_refused(tmp_path: Path) -> None:
    """``weights_only=True``: a matching digest is not a licence to unpickle.

    The digest only proves the bytes are the ones that were pinned. If the
    pinned artifact were itself hostile - a compromised Hub repo, a bad pin -
    the loader still must not execute it.
    """
    checkpoint = tmp_path / "malicious.pt"
    marker = tmp_path / "executed"
    torch.save(_UnsafeCheckpointPayload(marker), checkpoint)

    with pytest.raises(CalamariUnavailableError, match="unable to safely load"):
        run_calamari_transcribe(
            TRANSCRIBE_LINE.read_bytes(),
            checkpoint_path=checkpoint,
            artifact_sha256=_digest(checkpoint),
        )

    assert not marker.exists()


def test_torch_load_is_called_with_weights_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the flag itself: it is one keyword between safe and arbitrary code.

    Observed at the call, not read out of the file. The module docstring at the
    top of ``checkpoint.py`` contains the phrase ``weights_only=True`` in prose,
    so a source-substring assertion stayed green with the keyword removed from
    the call two dozen lines below it - the single mutation this test exists to
    catch.
    """
    real_load = torch.load
    calls: list[dict[str, object]] = []

    def spy(*args: object, **kwargs: object):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    # `checkpoint.py` does `import torch` and resolves the attribute at call
    # time, so the module's `torch` *is* this one.
    monkeypatch.setattr(torch, "load", spy)

    model, metadata = load_calamari_checkpoint(CALAMARI_CHECKPOINT)

    assert metadata.classes == 47
    assert model is not None
    assert len(calls) == 1, f"expected exactly one torch.load, saw {len(calls)}"
    assert calls[0].get("weights_only") is True
    # CPU only, for the same reason as `test_the_runtime_never_selects_an_accelerator`.
    assert calls[0].get("map_location") == "cpu"

    # One call site, so the observation above covers the module rather than one
    # branch of it. This is the only claim here the source can answer.
    source = (REPO_ROOT / "inference/architectures/calamari/checkpoint.py").read_text(
        encoding="utf-8"
    )
    assert source.count("torch.load(") == 1


def test_blla_never_unpickles_during_a_real_segment_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """BLLA's artifact format cannot execute code at all, and must stay that way.

    ``"torch.load" not in source`` was the old form of this, and moving the load
    one module over defeated it while changing nothing about what runs. So a
    whole page is segmented for real with ``torch.load`` replaced by something
    that cannot succeed: wherever the load lives now, this notices it.
    """
    from inference.architectures.blla.blla import _load_blla_model

    reached: list[tuple[object, ...]] = []

    def refuse(*args: object, **kwargs: object):
        reached.append(args)
        raise AssertionError("BLLA segmentation reached torch.load")

    # The model is cached per path; a warm cache would skip the load entirely
    # and make this pass without proving anything.
    _load_blla_model.cache_clear()
    monkeypatch.setattr(torch, "load", refuse)

    response = run_blla_segment(SEGMENT_PAGE.read_bytes(), model_path=BLLA_CHECKPOINT)

    assert reached == []
    assert len(response.lines) > 10, "the run has to actually load and execute the model"


def test_blla_refuses_a_pickled_checkpoint_outright() -> None:
    """Not merely "does not need pickle": will not open one when handed one."""
    with pytest.raises(BLLAUnavailableError):
        run_blla_segment(SEGMENT_PAGE.read_bytes(), model_path=CALAMARI_CHECKPOINT)


# --- The checkpoint the registry pins is the one the graph accepts ------------


def test_the_bundled_checkpoint_matches_the_runtime_graph_exactly() -> None:
    """``strict=True`` on load: a drifted graph fails here, not at request time."""
    model, metadata = load_calamari_checkpoint(CALAMARI_CHECKPOINT)

    assert metadata.classes == 47
    assert metadata.line_height == 48
    assert len(metadata.charset) == metadata.classes
    assert metadata.blank_index == 0
    assert not model.training


def test_the_registry_pins_the_digest_of_the_native_artifact() -> None:
    """The pinned digest must be the file the loader will actually open."""
    import yaml

    registry = yaml.safe_load((REPO_ROOT / "inference/registry.yaml").read_text(encoding="utf-8"))
    calamari = registry["models"]["syriac-calamari-v1"]["versions"]["stable"]

    # The offline bundled weights are the same bytes as the published best.pt.
    assert calamari["artifact_sha256"] == _digest(CALAMARI_CHECKPOINT)

    blla = registry["models"]["blla-segment"]["versions"]["stable"]
    assert blla["artifact_sha256"] == _digest(BLLA_CHECKPOINT)


def test_a_page_that_segments_can_be_transcribed_line_by_line() -> None:
    """The two architectures compose: segment output feeds transcribe input."""
    from inference.architectures.calamari.adapter import run_calamari_transcribe_many
    from inference.jobs.runner import _crop_line_image

    page = SEGMENT_PAGE.read_bytes()
    segmented = run_blla_segment(page, model_path=BLLA_CHECKPOINT)
    crops = [_crop_line_image(page, line.points) for line in segmented.lines[:5]]
    assert len(crops) == 5

    results = run_calamari_transcribe_many(crops, checkpoint_path=CALAMARI_CHECKPOINT)

    assert len(results) == 5
    assert all(hasattr(result, "text") for result in results)


def test_segment_geometry_stays_inside_the_page() -> None:
    with Image.open(SEGMENT_PAGE) as image:
        width, height = image.size

    response = run_blla_segment(SEGMENT_PAGE.read_bytes(), model_path=BLLA_CHECKPOINT)

    for line in response.lines:
        for x, y in line.points:
            assert -1 <= x <= width + 1
            assert -1 <= y <= height + 1
