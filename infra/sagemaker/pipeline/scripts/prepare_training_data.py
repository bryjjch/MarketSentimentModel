"""SageMaker Processing script: assemble training data for the FinSense pipeline.

Reads curated JSONL partitions (high-confidence model labels + pseudo-labels from the
daily ingestion pipeline) and Financial PhraseBank, then writes channel-ready outputs
for the downstream MLM and classifier training steps plus a held-out test split for
independent evaluation.

Processing I/O (set by the pipeline definition):
  Inputs:
    /opt/ml/processing/input/curated/   curated/*.jsonl from the data bucket
  Outputs:
    /opt/ml/processing/output/mlm_corpus/        unlabeled text for MLM pre-training
    /opt/ml/processing/output/phrasebank_train/  PhraseBank train split (original format)
    /opt/ml/processing/output/pseudo_data/       labeled JSONL for classifier --pseudo_data
    /opt/ml/processing/output/test_data/         held-out PhraseBank test split (JSONL)
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SENTIMENT_STR_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
FINPHRASE_ZIP_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/"
    "FinancialPhraseBank-v1.0.zip?download=true"
)


def _safe_zip_extractall(zf: zipfile.ZipFile, extract_dir: Path) -> None:
    """Extract a ZipFile while guarding against path-traversal entries."""
    extract_dir_resolved = extract_dir.resolve()
    for member in zf.infolist():
        member_path = (extract_dir_resolved / member.filename).resolve()
        if not member_path.is_relative_to(extract_dir_resolved):
            raise ValueError(f"Unsafe ZIP member path rejected: {member.filename!r}")
    zf.extractall(extract_dir)


def _download_phrasebank(dest_dir: Path, subset: str = "Sentences_75Agree.txt") -> Path:
    """Download and extract Financial PhraseBank; return path to the chosen subset file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "FinancialPhraseBank-v1.0.zip"
    extract_dir = dest_dir / "FinancialPhraseBank-v1.0"
    if not zip_path.is_file():
        print("Downloading Financial PhraseBank ...")
        urllib.request.urlretrieve(FINPHRASE_ZIP_URL, str(zip_path))
        print()
    if not extract_dir.is_dir():
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_zip_extractall(zf, extract_dir)
    inner = extract_dir / "FinancialPhraseBank-v1.0" / subset
    if not inner.is_file():
        raise FileNotFoundError(f"Expected {inner}")
    return inner


def _load_phrasebank(txt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        txt_path,
        sep="@",
        header=None,
        names=["text", "sentiment"],
        encoding="ISO-8859-1",
        engine="python",
    )
    df["label"] = df["sentiment"].str.strip().str.lower().map(SENTIMENT_STR_TO_ID)
    return df[["text", "label"]]


def _load_curated_jsonl(curated_dir: Path) -> pd.DataFrame:
    """Read all JSONL files under the curated input directory."""
    rows: list[dict] = []
    if not curated_dir.is_dir():
        return pd.DataFrame(columns=["text", "label"])
    for p in sorted(curated_dir.rglob("*.jsonl")):
        with p.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text")
                label = obj.get("label_id")
                if text and label is not None:
                    rows.append({"text": str(text).strip(), "label": int(label)})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["text", "label"])
    return df[["text", "label"]]


def _write_phrasebank_format(df: pd.DataFrame, dest: Path) -> None:
    """Write a DataFrame in PhraseBank's ``text@sentiment`` ``@``-delimited format."""
    id_to_str = {0: "negative", 1: "neutral", 2: "positive"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="ISO-8859-1") as f:
        for _, row in df.iterrows():
            f.write(f"{row['text']}@{id_to_str[int(row['label'])]}\n")


def _write_jsonl(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({"text": row["text"], "label": int(row["label"])}) + "\n")


def _write_text_corpus(texts: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phrasebank-subset", default="Sentences_75Agree.txt")
    args = parser.parse_args()

    input_curated = Path(os.environ.get("SM_CHANNEL_CURATED", "/opt/ml/processing/input/curated"))
    out_mlm = Path("/opt/ml/processing/output/mlm_corpus")
    out_pb_train = Path("/opt/ml/processing/output/phrasebank_train")
    out_pseudo = Path("/opt/ml/processing/output/pseudo_data")
    out_test = Path("/opt/ml/processing/output/test_data")

    # --- PhraseBank ---------------------------------------------------------
    pb_txt = _download_phrasebank(Path("/tmp/phrasebank"), subset=args.phrasebank_subset)
    df_pb = _load_phrasebank(pb_txt)
    print(f"PhraseBank rows: {len(df_pb)}")

    strat = df_pb["label"] if df_pb["label"].value_counts().min() >= 2 else None
    pb_train, pb_test = train_test_split(
        df_pb, test_size=args.test_ratio, random_state=args.seed, stratify=strat,
    )
    print(f"PhraseBank train: {len(pb_train)}, test: {len(pb_test)}")

    # --- Curated data (high-confidence + pseudo-labeled) --------------------
    df_curated = _load_curated_jsonl(input_curated)
    print(f"Curated rows loaded: {len(df_curated)}")

    # --- Write outputs ------------------------------------------------------
    # 1. MLM corpus: all available text (PhraseBank train + curated), unlabeled
    all_texts = pb_train["text"].tolist()
    if not df_curated.empty:
        all_texts.extend(df_curated["text"].tolist())
    _write_text_corpus(all_texts, out_mlm / "corpus.jsonl")
    print(f"MLM corpus: {len(all_texts)} texts -> {out_mlm / 'corpus.jsonl'}")

    # 2. PhraseBank train split (original @-delimited format for --phrasebank_txt)
    _write_phrasebank_format(pb_train, out_pb_train / args.phrasebank_subset)
    print(f"PhraseBank train split -> {out_pb_train / args.phrasebank_subset}")

    # 3. Pseudo / curated labeled data for --pseudo_data
    if not df_curated.empty:
        _write_jsonl(df_curated, out_pseudo / "pseudo.jsonl")
        print(f"Pseudo data: {len(df_curated)} rows -> {out_pseudo / 'pseudo.jsonl'}")
    else:
        _write_jsonl(pd.DataFrame(columns=["text", "label"]), out_pseudo / "pseudo.jsonl")
        print("No curated rows; wrote empty pseudo.jsonl placeholder")

    # 4. Test split (JSONL for the evaluation processing step)
    _write_jsonl(pb_test, out_test / "test.jsonl")
    print(f"Test split: {len(pb_test)} rows -> {out_test / 'test.jsonl'}")


if __name__ == "__main__":
    main()
