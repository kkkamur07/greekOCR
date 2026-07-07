"""Calamari adapter unwraps tfaip Sample outputs from predict_raw."""

from __future__ import annotations

from dataclasses import dataclass

from inference.architectures.calamari import _prediction_result, _response_from_prediction


@dataclass
class _FakePrediction:
    sentence: str
    avg_char_probability: float = 0.9
    positions: list = None

    def __post_init__(self) -> None:
        if self.positions is None:
            self.positions = []


@dataclass
class _FakeSample:
    outputs: _FakePrediction


def test_response_from_prediction_reads_sentence_on_sample_outputs() -> None:
    sample = _FakeSample(outputs=_FakePrediction(sentence="ܡܪܝ", avg_char_probability=0.81))
    response = _response_from_prediction(sample)
    assert response.text == "ܡܪܝ"
    assert response.confidence == 0.81


def test_prediction_result_passes_through_bare_prediction() -> None:
    prediction = _FakePrediction(sentence="alpha")
    assert _prediction_result(prediction) is prediction
