"""Shared helpers for the financial sentiment training pipeline."""

from __future__ import annotations

import math
import os
import shutil
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


def trainer_warmup_steps(
    *,
    num_train_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    warmup_ratio: float,
) -> int:
    """
    Optimizer warmup steps consistent with ``transformers.Trainer`` epoch scheduling.

    ``TrainingArguments.warmup_ratio`` is deprecated in favor of ``warmup_steps``; this
    mirrors ``Trainer.set_initial_training_values`` (dataloader length + grad accumulation).
    Per-device batch size and gradient accumulation are clamped to at least 1, matching
    Trainer's safe defaults when arguments are underspecified.
    """
    if warmup_ratio <= 0:
        return 0
    per_device_bs = max(1, int(per_device_train_batch_size))
    grad_accum = max(1, int(gradient_accumulation_steps))
    len_dataloader = max(1, math.ceil(num_train_examples / per_device_bs))
    num_update_steps_per_epoch = max(
        len_dataloader // grad_accum + int(len_dataloader % grad_accum > 0),
        1,
    )
    max_steps = math.ceil(float(num_train_epochs) * num_update_steps_per_epoch)
    return int(max_steps * warmup_ratio)


def default_data_root() -> Path:
    """Default directory for downloaded corpora: ``<repository root>/data``."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "data"


def _env_path(var_name: str) -> Path | None:
    """Return ``Path`` from a whitespace-trimmed environment variable, or ``None`` if unset/empty."""
    value = os.environ.get(var_name, "").strip()
    return Path(value) if value else None


def sagemaker_model_dir() -> Path | None:
    """Return ``SM_MODEL_DIR`` (auto-packaged into ``model.tar.gz`` at training job end) if set."""
    return _env_path("SM_MODEL_DIR")


def sagemaker_output_data_dir() -> Path | None:
    """Return ``SM_OUTPUT_DATA_DIR`` (uploaded to the job's output S3 path as ``output.tar.gz``) if set."""
    return _env_path("SM_OUTPUT_DATA_DIR")


def sagemaker_channel_dir(channel: str) -> Path | None:
    """Return ``SM_CHANNEL_<CHANNEL>`` directory (staged from S3 by SageMaker) if set."""
    return _env_path(f"SM_CHANNEL_{channel.upper()}")


def default_serving_code_dir() -> Path:
    """
    Return the default SageMaker serving ``code/`` directory when running from the repository.

    This default relies on the source tree layout (``<repo>/infra/sagemaker/serving/code``).
    In packaged installs, that directory may not be present, so callers must provide an
    explicit serving code directory instead of relying on the repo-relative fallback.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    serving_code_dir = repo_root / "infra" / "sagemaker" / "serving" / "code"
    if not serving_code_dir.is_dir():
        raise FileNotFoundError(
            "Default SageMaker serving code directory was not found at "
            f"{serving_code_dir}. This default only works when running from the project "
            "repository with infra/sagemaker/serving/code available. In packaged installs, "
            "provide an explicit serving code directory instead of relying on "
            "default_serving_code_dir()."
        )
    return serving_code_dir


def copy_serving_code_into(target_dir: str | Path, source_dir: str | Path) -> Path:
    """
    Copy SageMaker serving code into ``<target_dir>/code/`` so the auto-packaged
    ``model.tar.gz`` contains the inference entrypoint expected by the Hugging Face DLC.

    Any existing ``code/`` subtree under ``target_dir`` is replaced. ``__pycache__`` and
    ``*.pyc`` files are excluded.
    """
    src = Path(source_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Serving code directory not found: {src}")
    dst = Path(target_dir) / "code"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dst


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
    """Load PhraseBank from tab-separated ``@`` files using ISO-8859-1 (dataset encoding)."""
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
