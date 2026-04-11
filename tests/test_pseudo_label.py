"""Tests for pseudo-label normalization."""

from __future__ import annotations

from training.pseudo_label import normalize_label


def test_normalize_label_word():
    assert normalize_label("positive") == 2
    assert normalize_label("The sentiment is neutral.") == 1
    assert normalize_label("NEGATIVE outlook") == 0


def test_normalize_label_unknown():
    assert normalize_label("maybe bullish") is None
