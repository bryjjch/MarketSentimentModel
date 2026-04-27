"""SageMaker Processing script: assemble training data for the FinSense pipeline.

Reads curated JSONL partitions (high-confidence model labels + pseudo-labels from the
daily ingestion pipeline) and Financial PhraseBank (pre-uploaded to S3 and staged by
the pipeline as a ProcessingInput), then writes channel-ready outputs for the downstream
MLM and classifier training steps plus a held-out test split for independent evaluation.

Processing I/O (set by the pipeline definition):
  Inputs:
    /opt/ml/processing/input/curated/      curated/*.jsonl from the data bucket
    /opt/ml/processing/input/phrasebank/   Sentences_75Agree.txt from s3://data-bucket/reference/phrasebank/
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
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SENTIMENT_STR_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


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
    input_phrasebank = Path(os.environ.get("SM_CHANNEL_PHRASEBANK", "/opt/ml/processing/input/phrasebank"))
    out_mlm = Path("/opt/ml/processing/output/mlm_corpus")
    out_pb_train = Path("/opt/ml/processing/output/phrasebank_train")
    out_pseudo = Path("/opt/ml/processing/output/pseudo_data")
    out_test = Path("/opt/ml/processing/output/test_data")

    # --- PhraseBank ---------------------------------------------------------
    pb_txt = input_phrasebank / args.phrasebank_subset
    if not pb_txt.is_file():
        raise FileNotFoundError(
            f"PhraseBank file not found at {pb_txt}. Ensure it is uploaded to the data "
            f"bucket under reference/phrasebank/ and staged as a ProcessingInput."
        )
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
