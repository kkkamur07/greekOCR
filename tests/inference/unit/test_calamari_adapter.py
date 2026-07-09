"""Calamari adapter response helpers."""

from __future__ import annotations

import pytest
from inference.architectures.calamari.adapter import _response_from_decoded


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
