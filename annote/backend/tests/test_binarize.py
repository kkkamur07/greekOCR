"""Binarize processing step."""

import numpy as np
import pytest

from annote.services.processing.binarize import binarize
from annote.services.processing.pipeline import process


def test_process_binarize_step(monkeypatch):
    page = np.full((40, 80, 3), 200, dtype=np.uint8)
    segment = {
        "kind": "rectangle",
        "points": [[10, 10], [70, 10], [70, 30], [10, 30]],
    }
    calls: list[np.ndarray] = []

    def fake_binarize(image: np.ndarray) -> np.ndarray:
        calls.append(image.copy())
        out = image.copy()
        out[:, :] = [0, 0, 0]
        return out

    monkeypatch.setattr("annote.services.processing.pipeline.binarize", fake_binarize)

    result = process(page, segment, ["rectify", "binarize"])

    assert len(calls) == 1
    assert result.shape[2] == 3
    assert np.all(result == 0)


def test_binarize_requires_kraken():
    pytest.importorskip("kraken")
    image = np.full((20, 40, 3), 180, dtype=np.uint8)
    image[8:14, 10:30] = 30

    result = binarize(image)

    assert result.shape[:2] == image.shape[:2]
    assert len(np.unique(result)) > 1
