"""Text-recognition metrics without external runtime dependencies."""

from __future__ import annotations

from collections.abc import Sequence


def edit_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for reference_index, reference_value in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_value in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1]
                    + (reference_value != prediction_value),
                )
            )
        previous = current
    return previous[-1]


def compute_text_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length.")
    character_errors = sum(edit_distance(reference, prediction) for prediction, reference in zip(predictions, references, strict=True))
    characters = sum(len(reference) for reference in references)
    word_errors = sum(
        edit_distance(reference.split(), prediction.split())
        for prediction, reference in zip(predictions, references, strict=True)
    )
    words = sum(len(reference.split()) for reference in references)
    exact_matches = sum(
        prediction == reference for prediction, reference in zip(predictions, references, strict=True)
    )
    return {
        "cer": character_errors / max(characters, 1),
        "wer": word_errors / max(words, 1),
        "exact_match": exact_matches / max(len(references), 1),
    }
