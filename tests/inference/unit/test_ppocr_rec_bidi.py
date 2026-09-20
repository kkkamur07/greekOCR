"""PP-OCR recognition bidi: the vendored reorder and the adapter's use of it.

The network emits characters in display order (left to right on the image);
kraken's ``BaselineOCRRecord.logical_order`` runs that string through its bidi
``get_display_map`` and permutes the confidences with the same map. These
tests pin the vendored copy's behavior on the cases serving depends on: a
Syriac display string comes back in logical order with a bijective
permutation, and a Latin string passes through unchanged.
"""

from __future__ import annotations

from nomikos_inference.architectures.ppocr_rec.adapter import _reorder_to_logical
from nomikos_inference.architectures.ppocr_rec.bidi import UCD_VERSION, get_display_map


def test_syriac_display_order_becomes_logical_order() -> None:
    text, order = get_display_map("ܝܪܡ", None)
    assert text == "ܡܪܝ"
    assert order == [2, 1, 0]


def test_permutation_is_a_bijection() -> None:
    text, order = get_display_map("ܝܪܡ", None)
    assert sorted(order) == list(range(len("ܝܪܡ")))
    assert len(text) == len(order)


def test_latin_string_is_unchanged() -> None:
    text, order = get_display_map("hello", None)
    assert text == "hello"
    assert order == [0, 1, 2, 3, 4]


def test_empty_string_reorders_to_empty() -> None:
    assert get_display_map("", None) == ("", [])


def test_adapter_reorder_permutes_confidences_with_the_text() -> None:
    text, confidences = _reorder_to_logical("ܝܪܡ", [0.1, 0.2, 0.3])
    assert text == "ܡܪܝ"
    assert confidences == [0.3, 0.2, 0.1]


def test_adapter_reorder_leaves_latin_confidences_in_place() -> None:
    text, confidences = _reorder_to_logical("ab", [0.5, 0.6])
    assert text == "ab"
    assert confidences == [0.5, 0.6]


def test_vendored_tables_match_the_pinned_unicode_version() -> None:
    assert UCD_VERSION == "17.0.0"
