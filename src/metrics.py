"""Shared framework-independent OCR text metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any


def edit_distance(reference: Sequence[Any], prediction: Sequence[Any]) -> int:
    """Return the Levenshtein distance between two sequences."""
    previous_row = list(range(len(prediction) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current_row = [reference_index]
        for prediction_index, prediction_item in enumerate(prediction, start=1):
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[prediction_index] + 1,
                    previous_row[prediction_index - 1]
                    + (reference_item != prediction_item),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def _words(text: str) -> list[str]:
    return text.split()


def compute_text_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, float]:
    """Compute corpus CER, WER, exact-match, and SROIE metrics."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length.")
    if not references:
        return {
            "cer": 0.0,
            "wer": 0.0,
            "exact_match": 0.0,
            "sroie_precision": 0.0,
            "sroie_recall": 0.0,
            "sroie_f1": 0.0,
        }

    character_edits = 0
    reference_characters = 0
    word_edits = 0
    reference_words = 0
    exact_matches = 0
    matched_sroie_words = 0
    predicted_sroie_words = 0
    reference_sroie_words = 0

    for reference, prediction in zip(references, predictions, strict=True):
        character_edits += edit_distance(reference, prediction)
        reference_characters += len(reference)
        exact_matches += int(reference == prediction)

        reference_tokens = _words(reference)
        prediction_tokens = _words(prediction)
        word_edits += edit_distance(reference_tokens, prediction_tokens)
        reference_words += len(reference_tokens)

        reference_counter = Counter(reference_tokens)
        prediction_counter = Counter(prediction_tokens)
        matched_sroie_words += sum(
            (reference_counter & prediction_counter).values()
        )
        reference_sroie_words += len(reference_tokens)
        predicted_sroie_words += len(prediction_tokens)

    cer = (
        character_edits / reference_characters
        if reference_characters
        else 0.0
    )
    wer = word_edits / reference_words if reference_words else 0.0
    exact_match = exact_matches / len(references)
    sroie_precision = (
        matched_sroie_words / predicted_sroie_words
        if predicted_sroie_words
        else 0.0
    )
    sroie_recall = (
        matched_sroie_words / reference_sroie_words
        if reference_sroie_words
        else 0.0
    )
    sroie_f1 = (
        2 * sroie_precision * sroie_recall / (sroie_precision + sroie_recall)
        if sroie_precision + sroie_recall
        else 0.0
    )
    return {
        "cer": cer,
        "wer": wer,
        "exact_match": exact_match,
        "sroie_precision": sroie_precision,
        "sroie_recall": sroie_recall,
        "sroie_f1": sroie_f1,
    }
