"""Tests for shared data helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from training.common import load_finphrasebank_dataframe, load_labeled_table


def test_load_finphrasebank_dataframe(tmp_path: Path) -> None:
    txt = tmp_path / "pb.txt"
    txt.write_text(
        'revenue beat expectations@positive\nlitigation costs rose@negative\ncompany issued guidance@neutral\n',
        encoding="ISO-8859-1",
    )
    df = load_finphrasebank_dataframe(txt)
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 3
    assert set(df["label"].tolist()) == {0, 1, 2}


def test_load_labeled_table_csv(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    p.write_text("text,label\nhello,0\nworld,2\n", encoding="utf-8")
    df = load_labeled_table(p)
    assert len(df) == 2
    assert df["label"].tolist() == [0, 2]


def test_load_labeled_table_jsonl_bad_label(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"text":"a","label":9}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Labels must be"):
        load_labeled_table(p)
