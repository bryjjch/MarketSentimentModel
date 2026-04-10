"""Shared helpers for the financial sentiment training pipeline."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd
import wget

# Align with Financial PhraseBank strings and inference convention (negative=0, neutral=1, positive=2).
SENTIMENT_STR_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
SENTIMENT_ID_TO_STR = {v: k for k, v in SENTIMENT_STR_TO_ID.items()}

FINPHRASE_ZIP_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/"
    "FinancialPhraseBank-v1.0.zip?download=true"
)


def default_data_root() -> Path:
    """Default directory for downloaded corpora: ``<repository root>/data``."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "data"


def ensure_finphrasebank(
    data_root: Path | None = None,
    subset: str = "Sentences_75Agree.txt",
) -> Path:
    """
    Download and extract Financial PhraseBank if missing. Returns path to the chosen .txt file.
    """
    root = Path(data_root) if data_root is not None else default_data_root()
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "FinancialPhraseBank-v1.0.zip"
    extract_dir = root / "FinancialPhraseBank-v1.0"

    if not zip_path.is_file():
        print("Downloading Financial PhraseBank...")
        wget.download(str(FINPHRASE_ZIP_URL), str(zip_path))
        print()

    if not extract_dir.is_dir():
        print("Extracting Financial PhraseBank...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    inner = extract_dir / "FinancialPhraseBank-v1.0" / subset
    if not inner.is_file():
        raise FileNotFoundError(f"Expected labeled file at {inner}")
    return inner


def load_finphrasebank_dataframe(txt_path: Path | None = None) -> pd.DataFrame:
    """Load the Financial PhraseBank data into a pandas dataframe"""
    path = txt_path or ensure_finphrasebank()
    df = pd.read_csv(
        path,
        sep="@",
        header=None,
        names=["text", "sentiment"],
        encoding="ISO-8859-1",
        engine="python",
    )
    df["label"] = df["sentiment"].str.strip().str.lower().map(SENTIMENT_STR_TO_ID)
    if df["label"].isna().any():
        bad = df.loc[df["label"].isna(), "sentiment"].unique()
        raise ValueError(f"Unknown sentiment labels in PhraseBank: {bad}")
    return df[["text", "label"]]


def load_labeled_table(path: Path, text_column: str = "text", label_column: str = "label") -> pd.DataFrame:
    """Load CSV or JSONL with at least text + integer label columns (0=neg, 1=neu, 2=pos)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix} (use .csv or .jsonl)")

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Expected columns {text_column!r} and {label_column!r}; got {list(df.columns)}")

    out = df[[text_column, label_column]].copy()
    out.columns = ["text", "label"]
    out["label"] = out["label"].astype(int)
    invalid = ~out["label"].isin(SENTIMENT_ID_TO_STR.keys())
    if invalid.any():
        raise ValueError(f"Labels must be in {set(SENTIMENT_ID_TO_STR.keys())}")
    return out
