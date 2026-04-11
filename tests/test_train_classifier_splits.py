"""Tests for train/val/test splitting."""

from __future__ import annotations

import pandas as pd

from training.train_classifier import split_labeled_frame


def test_split_no_test_matches_legacy_fractions():
    df = pd.DataFrame(
        {
            "text": [f"x{i}" for i in range(20)],
            "label": [0, 1, 2] * 6 + [0, 1],
        }
    )
    train_df, val_df, test_df = split_labeled_frame(df, val_ratio=0.25, test_ratio=0.0, seed=0)
    assert test_df is None
    assert len(train_df) + len(val_df) == 20
    assert len(val_df) == 5


def test_split_with_test_holdout():
    df = pd.DataFrame(
        {
            "text": [f"x{i}" for i in range(30)],
            "label": [0, 1, 2] * 10,
        }
    )
    train_df, val_df, test_df = split_labeled_frame(df, val_ratio=0.2, test_ratio=0.1, seed=42)
    assert test_df is not None
    assert len(test_df) == 3
    assert len(train_df) + len(val_df) + len(test_df) == 30
    assert set(train_df.index).isdisjoint(test_df.index)
    assert set(val_df.index).isdisjoint(test_df.index)
