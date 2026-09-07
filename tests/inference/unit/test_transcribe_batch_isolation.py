"""Per-line error isolation for the batched transcribe path.

One malformed line on a forty-line page must cost that line only. These tests
pin both halves of that: the surviving lines still come back with text, and a
batch where nothing survived is still a failure rather than an empty success.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from nomikos_inference.architectures.calamari import adapter
from nomikos_inference.architectures.calamari.adapter import TranscribeLineFailure
from nomikos_inference.contracts.common import InferenceTask, LineCrop, RegistryArchitecture
from nomikos_inference.contracts.transcribe import (
    TRANSCRIBE_LINE_ERROR,
    CharacterConfidence,
    TranscribeBatchLineResult,
    TranscribeBatchRunResponse,
    TranscribeRunResponse,
)
from nomikos_inference.jobs.runner import run_model
from tests.fixtures.paths import TRANSCRIBE_LINE

# The published graph, from the Hub revision the registry pins. Gitignored, so
# these two real-weights cases skip in a checkout that has not fetched it; every
# other case here injects failures at the batch seam and needs no artifact.
CALAMARI_ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "nomikos_inference/publish/artifacts/cache/syriac-calamari-v2/stable/best.onnx"
)
requires_artifact = pytest.mark.skipif(
    not CALAMARI_ARTIFACT.is_file(), reason="published Calamari artifact is not cached locally"
)


def _png_line(width: int = 40) -> bytes:
    output = BytesIO()
    Image.new("L", (width, 12), 255).save(output, format="PNG")
    return output.getvalue()


def _transcribed(text: str) -> TranscribeRunResponse:
    return TranscribeRunResponse(
        text=text,
        confidence=0.9,
        character_confidences=[CharacterConfidence(char=char, confidence=0.9) for char in text],
    )


def _line_params(count: int) -> dict:
    return {
        "lines": [
            {
                "line_id": f"line-{index}",
                "line_index": index,
                "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
            }
            for index in range(count)
        ]
    }


@pytest.fixture
def calamari_runner(monkeypatch: pytest.MonkeyPatch):
    """Wire ``run_model`` to a Calamari entry without touching weights.

    ``line_crop`` and ``line_crop_padding`` are on the stub entry because the
    registry requires both on every transcribe model: how a model's training
    crops were cut is per model, and the runner reads it off the resolved entry
    (ADR 0007).
    """
    monkeypatch.setattr("nomikos_inference.jobs.runner.validate_image_bytes", lambda *_args: None)
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.validate_request_params", lambda *_args: None
    )
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.get_inference_settings",
        lambda: SimpleNamespace(inference_registry_path=Path("registry.yaml")),
    )
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.resolve_registry_entry",
        lambda **_kwargs: SimpleNamespace(
            architecture=RegistryArchitecture.calamari,
            line_crop=LineCrop.polygon_white,
            line_crop_padding=12,
            versions={
                "stable": SimpleNamespace(
                    weights_source="file://unused",
                    hub_revision=None,
                    artifact_sha256=None,
                )
            },
        ),
    )
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.resolve_weights_source",
        lambda *_args, **_kwargs: CALAMARI_ARTIFACT,
    )

    def run(params: dict, image_bytes: bytes | None = None):
        return run_model(
            task=InferenceTask.transcribe,
            registry_model_id="model",
            registry_tag="stable",
            image_bytes=image_bytes if image_bytes is not None else _png_line(),
            params=params,
        )

    return run


def test_one_failing_line_still_returns_the_other_lines(
    calamari_runner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.run_calamari_transcribe_many",
        lambda line_images, **_kwargs: [
            _transcribed("a"),
            TranscribeLineFailure(index=1, error=ValueError("undecodable crop")),
            _transcribed("c"),
        ],
    )

    output = calamari_runner(_line_params(3))

    assert isinstance(output, TranscribeBatchRunResponse)
    assert [line.line_index for line in output.lines] == [0, 1, 2]
    assert [line.line_id for line in output.lines] == ["line-0", "line-1", "line-2"]
    assert output.lines[0].output is not None and output.lines[0].output.text == "a"
    assert output.lines[2].output is not None and output.lines[2].output.text == "c"
    # The failed line carries an error in place of its output, and the static
    # message never leaks the underlying exception.
    assert output.lines[1].output is None
    assert output.lines[1].error == TRANSCRIBE_LINE_ERROR
    assert "undecodable" not in TRANSCRIBE_LINE_ERROR


def test_uncroppable_line_geometry_is_isolated_from_the_batch(
    calamari_runner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crop(
        image,
        image_bytes: bytes,
        points: list[list[float]] | None,
        line_crop: LineCrop = LineCrop.polygon_white,
        line_crop_padding: int = 12,
    ) -> bytes:
        if points and points[0][0] == 99.0:
            raise ValueError("degenerate polygon")
        return image_bytes

    monkeypatch.setattr("nomikos_inference.jobs.runner._crop_line_image", crop)
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.run_calamari_transcribe_many",
        lambda line_images, **_kwargs: [_transcribed("a") for _ in line_images],
    )
    params = _line_params(3)
    params["lines"][1]["points"] = [[99.0, 0.0], [99.0, 0.0], [99.0, 0.0], [99.0, 0.0]]

    output = calamari_runner(params)

    # The bad region never reaches the model, and the good ones keep their
    # position: the line at index 2 must not slide into the failed line's slot.
    assert output.lines[1].error == TRANSCRIBE_LINE_ERROR
    assert output.lines[0].output is not None
    assert output.lines[2].output is not None
    assert output.lines[2].line_id == "line-2"


def test_batch_with_no_croppable_line_fails_instead_of_returning_nothing(
    calamari_runner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner._crop_line_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("degenerate polygon")),
    )
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.run_calamari_transcribe_many",
        lambda *_args, **_kwargs: pytest.fail("the model must not run without a single crop"),
    )

    with pytest.raises(ValueError, match="no transcribable line regions"):
        calamari_runner(_line_params(3))


@requires_artifact
def test_one_undecodable_crop_does_not_take_the_page_down() -> None:
    """Run the real checkpoint: one bad crop, two real ones, one page.

    No stubbed session. The middle "crop" is not an image at all, so
    preprocessing raises inside the batch loop and the adapter has to isolate
    it while the two real line crops still transcribe.
    """
    results = adapter.run_calamari_transcribe_many(
        [TRANSCRIBE_LINE.read_bytes(), b"not an image", TRANSCRIBE_LINE.read_bytes()],
        checkpoint_path=CALAMARI_ARTIFACT,
    )

    assert len(results) == 3
    assert isinstance(results[1], TranscribeLineFailure)
    assert results[1].index == 1
    survivors = [result for result in results if isinstance(result, TranscribeRunResponse)]
    assert len(survivors) == 2
    # The surviving lines produced real text from real weights, not a placeholder.
    assert all(survivor.text for survivor in survivors)
    assert survivors[0].text == survivors[1].text


@requires_artifact
def test_batch_where_every_line_failed_reraises_the_original_error() -> None:
    """An all-failed batch must keep the cause, not return an empty page.

    Run live against the real checkpoint: every crop is unreadable, so every
    line fails and the first failure comes back out with its original type. The
    type matters because it is what the agent reports as the reason - here it is
    PIL's ``UnidentifiedImageError`` rather than a generic wrapper, and an empty
    successful page would have been the far worse answer.
    """
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        adapter.run_calamari_transcribe_many(
            [b"not an image", b"also not an image"],
            checkpoint_path=CALAMARI_ARTIFACT,
        )


def test_line_result_requires_exactly_one_of_output_or_error() -> None:
    # The old, output-only shape still validates unchanged.
    assert TranscribeBatchLineResult(line_index=0, output=_transcribed("a")).error is None

    with pytest.raises(ValidationError, match="exactly one of output or error"):
        TranscribeBatchLineResult(line_index=0)
    with pytest.raises(ValidationError, match="exactly one of output or error"):
        TranscribeBatchLineResult(line_index=0, output=_transcribed("a"), error="boom")


def test_batch_response_rejects_an_all_error_body() -> None:
    with pytest.raises(ValidationError, match="at least one transcribed line"):
        TranscribeBatchRunResponse(
            lines=[
                TranscribeBatchLineResult(line_index=0, error=TRANSCRIBE_LINE_ERROR),
                TranscribeBatchLineResult(line_index=1, error=TRANSCRIBE_LINE_ERROR),
            ]
        )


@pytest.mark.parametrize("padding", [0, 12])
def test_the_registry_crop_settings_reach_the_crop(
    calamari_runner,
    monkeypatch: pytest.MonkeyPatch,
    padding: int,
) -> None:
    """The model's own crop settings are what the page is actually cut with.

    Greek and Armenian were trained by the same team on the same architecture and
    still need different paddings, so a hard-coded number here reads one of them
    wrong every time and never raises: Greek at Armenian's 12 px drops from 125
    exact lines out of 204 to 0. Pin the wiring rather than the pixels; the crop
    itself is compared against the training function in
    ``test_calamari_training_parity.py``.
    """
    import numpy as np

    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.resolve_registry_entry",
        lambda **_kwargs: SimpleNamespace(
            architecture=RegistryArchitecture.calamari,
            line_crop=LineCrop.polygon_white,
            line_crop_padding=padding,
            versions={
                "stable": SimpleNamespace(
                    weights_source="file://unused",
                    hub_revision=None,
                    artifact_sha256=None,
                )
            },
        ),
    )
    seen: list[tuple[LineCrop, int]] = []

    def record(page, points, convention: LineCrop, *, padding: int):
        seen.append((convention, padding))
        return np.zeros((6, 12), dtype=np.uint8)

    monkeypatch.setattr("nomikos_inference.jobs.runner.crop_line", record)
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.run_calamari_transcribe_many",
        lambda line_images, **_kwargs: [_transcribed("a") for _ in line_images],
    )

    calamari_runner(_line_params(2))

    assert seen == [(LineCrop.polygon_white, padding)] * 2
