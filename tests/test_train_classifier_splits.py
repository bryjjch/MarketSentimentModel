"""Tests for train/val/test splitting."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from training.train_classifier import (
    model_uses_finbert_label_order,
    phrasebank_labels_to_finbert,
    prepare_train_val_test_dataframes,
    split_labeled_frame,
)


def _phrasebank_order_model():
    return SimpleNamespace(
        config=SimpleNamespace(
            id2label={0: "negative", 1: "neutral", 2: "positive"},
        )
    )


def _finbert_order_model():
    return SimpleNamespace(
        config=SimpleNamespace(
            id2label={0: "positive", 1: "negative", 2: "neutral"},
        )
    )


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


def test_pseudo_rows_never_in_val_or_test():
    df_pb = pd.DataFrame(
        {
            "text": [f"pb{i}" for i in range(30)],
            "label": [0, 1, 2] * 10,
        }
    )
    df_pseudo = pd.DataFrame(
        {
            "text": [f"pseudo_only_{i}" for i in range(5)],
            "label": [0, 1, 2, 0, 1],
        }
    )
    pseudo_texts = set(df_pseudo["text"])
    train_df, val_df, test_df = prepare_train_val_test_dataframes(
        df_pb,
        df_pseudo,
        model=_phrasebank_order_model(),
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    )
    assert pseudo_texts.isdisjoint(set(val_df["text"]))
    assert test_df is not None
    assert pseudo_texts.isdisjoint(set(test_df["text"]))
    assert pseudo_texts <= set(train_df["text"])
    assert len(train_df) == len(df_pb) - len(val_df) - len(test_df) + len(df_pseudo)


def test_finbert_label_order_applied_to_phrasebank_and_pseudo_consistently():
    assert model_uses_finbert_label_order(_finbert_order_model())
    df_pb = pd.DataFrame(
        {
            "text": [f"pb{i}" for i in range(30)],
            "label": [0, 1, 2] * 10,
        }
    )
    df_pseudo = pd.DataFrame(
        {
            "text": [f"pseudo_only_{i}" for i in range(4)],
            "label": [2, 0, 1, 0],
        }
    )
    val_ratio = 0.2
    test_ratio = 0.1
    seed = 7

    train_df, val_df, test_df = prepare_train_val_test_dataframes(
        df_pb,
        df_pseudo,
        model=_finbert_order_model(),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    _, val_ref, test_ref = split_labeled_frame(
        phrasebank_labels_to_finbert(df_pb.copy()),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    pd.testing.assert_frame_equal(val_df.reset_index(drop=True), val_ref.reset_index(drop=True))
    assert test_df is not None and test_ref is not None
    pd.testing.assert_frame_equal(test_df.reset_index(drop=True), test_ref.reset_index(drop=True))

    pseudo_expected = phrasebank_labels_to_finbert(df_pseudo.copy()).set_index("text")
    got_pseudo = train_df[train_df["text"].isin(df_pseudo["text"])].set_index("text").sort_index()
    pd.testing.assert_series_equal(
        got_pseudo["label"].sort_index(),
        pseudo_expected["label"].sort_index(),
        check_names=False,
    )

    train_pb_only = train_df[train_df["text"].isin(df_pb["text"])].sort_values("text")
    val_texts = set(val_ref["text"])
    test_texts = set(test_ref["text"])
    assert set(train_pb_only["text"]).isdisjoint(val_texts | test_texts)
    train_pb_expected = phrasebank_labels_to_finbert(df_pb.copy())
    train_pb_expected = train_pb_expected[
        ~train_pb_expected["text"].isin(val_texts | test_texts)
    ].sort_values("text")
    pd.testing.assert_frame_equal(
        train_pb_only.reset_index(drop=True),
        train_pb_expected.reset_index(drop=True),
    )
