"""Calamari adapter response helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from inference.architectures.calamari.adapter import (  # noqa: E402
    CalamariUnavailableError,
    _load_checkpoint,
    _response_from_decoded,
)


def _write_marker(marker_path: str) -> None:
    Path(marker_path).write_text("unsafe checkpoint deserialized")


class _UnsafeCheckpointPayload:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _write_marker, (str(self.marker_path),)


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


def test_load_checkpoint_rejects_pickle_payload_without_executing_it(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "malicious.pt"
    marker_path = tmp_path / "executed"
    torch.save(_UnsafeCheckpointPayload(marker_path), checkpoint_path)

    with pytest.raises(CalamariUnavailableError, match="unable to safely load"):
        _load_checkpoint(str(checkpoint_path))

    assert not marker_path.exists()


def test_load_checkpoint_rejects_digest_mismatch_before_deserialization(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "malicious.pt"
    marker_path = tmp_path / "executed"
    torch.save(_UnsafeCheckpointPayload(marker_path), checkpoint_path)

    from inference.architectures.calamari.adapter import run_calamari_transcribe

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        run_calamari_transcribe(
            b"not-read-after-integrity-failure",
            checkpoint_path=checkpoint_path,
            artifact_sha256="0" * 64,
        )

    assert not marker_path.exists()


def test_load_checkpoint_rejects_incompatible_state_dictionary(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "incompatible.pt"
    torch.save(
        {
            "format": "calamari-pytorch-v1",
            "classes": 2,
            "line_height": 48,
            "charset": ["", "a"],
            "state_dict": {"unexpected.weight": torch.zeros(1)},
        },
        checkpoint_path,
    )

    with pytest.raises(CalamariUnavailableError, match="state dictionary is incompatible"):
        _load_checkpoint(str(checkpoint_path))
