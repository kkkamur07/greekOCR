"""Calamari adapter response helpers and artifact preflight.

Torch-free by construction, like everything under ``tests/inference``: the
published wheel cannot import Torch under ADR 0006, so a test that needs it
belongs in ``tests/export`` beside the graph it exercises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.architectures.calamari.adapter import (
    CalamariUnavailableError,
    _load_session,
    _response_from_decoded,
    run_calamari_transcribe,
    run_calamari_transcribe_many,
)


def test_response_from_decoded_aligns_character_confidences() -> None:
    response = _response_from_decoded("ܡܪܝ", [0.8, 0.9, 0.7])
    assert response.text == "ܡܪܝ"
    assert response.confidence == pytest.approx(0.8)
    assert [entry.char for entry in response.character_confidences] == ["ܡ", "ܪ", "ܝ"]
    assert [entry.confidence for entry in response.character_confidences] == [0.8, 0.9, 0.7]


def test_response_from_decoded_fills_missing_confidences() -> None:
    response = _response_from_decoded("ab", [0.5])
    assert response.text == "ab"
    assert [entry.confidence for entry in response.character_confidences] == [0.5, 0.5]


def test_load_session_rejects_a_file_that_is_not_an_onnx_graph(tmp_path: Path) -> None:
    artifact = tmp_path / "corrupt.onnx"
    artifact.write_bytes(b"not a protobuf")

    with pytest.raises(CalamariUnavailableError, match="unable to load Calamari ONNX artifact"):
        _load_session(str(artifact))


def test_a_native_checkpoint_is_refused_rather_than_loaded(tmp_path: Path) -> None:
    """The runtime accepts one format, and ``.pt`` is not it.

    Under ADR 0004 this path ran ``torch.load`` and the digest check was what
    kept an unverified pickle out of it. There is no unpickling left to reach:
    the suffix check refuses the file first, which is a stronger position than
    the one the digest was defending.
    """
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"pickled payload")

    with pytest.raises(CalamariUnavailableError, match="requires an .onnx model"):
        run_calamari_transcribe(b"unread", checkpoint_path=checkpoint_path)


def test_digest_is_verified_before_the_artifact_is_opened(tmp_path: Path) -> None:
    artifact = tmp_path / "best.onnx"
    artifact.write_bytes(b"not a protobuf either")

    # A SHA-256 mismatch, not a parse error: the digest is checked first, so a
    # tampered artifact is reported as an integrity failure rather than as
    # whatever onnxruntime happens to say about the bytes.
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        run_calamari_transcribe(
            b"not-read-after-integrity-failure",
            checkpoint_path=artifact,
            artifact_sha256="0" * 64,
        )


def test_empty_batch_is_a_client_error_even_when_the_weights_are_missing(tmp_path: Path) -> None:
    """422 beats 503: the request was unrunnable whatever is on disk.

    Ported from ``fix/remediation-runtime``, which fixed this ordering against
    the Torch adapter. ADR 0006 restored the ONNX adapter out of the archive and
    reintroduced the original ordering with it, so the guard is re-asserted here
    against the ONNX path rather than lost with the Torch one.
    """
    with pytest.raises(ValueError, match="at least one line image") as caught:
        run_calamari_transcribe_many([], checkpoint_path=tmp_path / "absent.onnx")

    assert not isinstance(caught.value, OSError)


def test_a_real_batch_still_reports_a_missing_artifact_first(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Calamari model not found"):
        run_calamari_transcribe_many([b"png"], checkpoint_path=tmp_path / "absent.onnx")
