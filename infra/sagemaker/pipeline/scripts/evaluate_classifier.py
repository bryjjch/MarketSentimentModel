"""SageMaker Processing script: evaluate the trained classifier against a held-out test set.

Loads the classifier model artifact, runs inference on test data, computes metrics, and
writes ``evaluation.json`` in the format expected by the pipeline's ConditionStep
(PropertyFile + JsonGet).

Processing I/O:
  Inputs:
    /opt/ml/processing/input/model/   classifier model.tar.gz (auto-extracted or raw)
    /opt/ml/processing/input/test/    test.jsonl from the data-prep step
  Outputs:
    /opt/ml/processing/output/evaluation/evaluation.json
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.evaluation import classification_metrics


def _safe_extractall(tar: tarfile.TarFile, extract_to: Path) -> None:
    """Extract all tar members, rejecting any paths that escape *extract_to*."""
    abs_dest = extract_to.resolve()
    for member in tar.getmembers():
        member_path = (abs_dest / member.name).resolve()
        if not str(member_path).startswith(str(abs_dest) + os.sep) and member_path != abs_dest:
            raise ValueError(f"Refusing unsafe tar member path: {member.name!r}")
    tar.extractall(extract_to)


def _extract_model(model_dir: Path) -> Path:
    """If model_dir contains a tar.gz instead of extracted weights, extract it first."""
    tar_files = list(model_dir.glob("*.tar.gz"))
    if tar_files:
        extract_to = model_dir / "_extracted"
        extract_to.mkdir(exist_ok=True)
        for tf in tar_files:
            with tarfile.open(tf, "r:gz") as tar:
                _safe_extractall(tar, extract_to)
        return extract_to

    if (model_dir / "config.json").exists():
        return model_dir

    for child in model_dir.iterdir():
        if child.is_dir() and (child / "config.json").exists():
            return child

    return model_dir


def _load_test_data(test_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(test_dir.rglob("*.jsonl")):
        with p.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> None:
    model_dir = Path(os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/processing/input/model"))
    test_dir = Path(os.environ.get("SM_CHANNEL_TEST", "/opt/ml/processing/input/test"))
    output_dir = Path("/opt/ml/processing/output/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = _extract_model(model_dir)
    print(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = _load_test_data(test_dir)
    if df.empty:
        raise ValueError("No test data found")
    print(f"Test samples: {len(df)}")

    texts = df["text"].tolist()
    labels = df["label"].astype(int).values

    batch_size = 32
    all_preds: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, max_length=175, padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)

    y_pred = np.array(all_preds)
    metrics = classification_metrics(labels, y_pred)

    # SageMaker Pipelines PropertyFile-compatible format
    evaluation = {
        "metrics": {
            name: {"value": value} for name, value in metrics.items()
        },
    }

    eval_path = output_dir / "evaluation.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    print(f"Evaluation written to {eval_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
