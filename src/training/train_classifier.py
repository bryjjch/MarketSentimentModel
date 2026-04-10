"""
Fine-tune a BERT-family classifier on Financial PhraseBank (and optional pseudo-labeled data).

Supports:
- FinBERT or other HF checkpoints (--base_model ProsusAI/finbert)
- Optional encoder weights from MLM continued pre-training (--mlm_checkpoint)
- Merging pseudo-labeled JSONL/CSV (--pseudo_data)

Saves a Hugging Face model directory compatible with my_model/code/inference.py.

Example:
  python -m training.train_classifier --base_model ProsusAI/finbert --output_dir outputs/clf_finbert
  python -m training.train_classifier --base_model bert-base-uncased --mlm_checkpoint outputs/mlm_bert --pseudo_data data/pseudo.jsonl --output_dir outputs/clf_mlm
  ms-train-classifier --base_model ProsusAI/finbert --output_dir outputs/clf_finbert
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .common import ensure_finphrasebank, load_finphrasebank_dataframe, load_labeled_table

# Financial PhraseBank / deployment label names.
PHRASEBANK_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# ProsusAI FinBERT uses 0=positive, 1=negative, 2=neutral.
PHRASEBANK_TO_FINBERT_LABEL = {0: 1, 1: 2, 2: 0}
_FINBERT_ROW_PERMUTE = [1, 2, 0]  # phrasebank row i <- finbert row perm[i]


def _classifier_linear(model: torch.nn.Module) -> torch.nn.Linear:
    """Get the linear classification head from the model"""
    head = getattr(model, "classifier", None)
    if isinstance(head, torch.nn.Linear):
        return head
    if head is not None and isinstance(head, torch.nn.Sequential):
        last = head[-1]
        if isinstance(last, torch.nn.Linear):
            return last
    raise ValueError("Could not find a Linear classification head on this model")


def model_uses_finbert_label_order(model) -> bool:
    """Check if the model uses the FinBERT label order"""
    id2 = model.config.id2label or {}
    if len(id2) != 3:
        return False

    def name(i: int) -> str:
        """Get the label name for a given index"""
        v = id2.get(i, id2.get(str(i)))
        return str(v).lower() if v is not None else ""

    return name(0) == "positive" and name(1) == "negative" and name(2) == "neutral"


def phrasebank_labels_to_finbert(df: pd.DataFrame) -> pd.DataFrame:
    """Convert PhraseBank labels to FinBERT labels"""
    out = df.copy()
    out["label"] = out["label"].map(PHRASEBANK_TO_FINBERT_LABEL)
    return out


def permute_classifier_to_phrasebank(model) -> None:
    """Permute the classifier to PhraseBank labels"""
    linear = _classifier_linear(model)
    perm = torch.tensor(_FINBERT_ROW_PERMUTE, dtype=torch.long)
    linear.weight.data = linear.weight.data[perm]
    if linear.bias is not None:
        linear.bias.data = linear.bias.data[perm]
    model.config.id2label = {i: PHRASEBANK_ID2LABEL[i] for i in range(3)}
    model.config.label2id = {PHRASEBANK_ID2LABEL[i]: i for i in range(3)}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    p = argparse.ArgumentParser(description="Financial sentiment classifier fine-tuning")
    p.add_argument(
        "--base_model",
        default="bert-base-uncased",
        help="HF model id or local path (e.g. ProsusAI/finbert, bert-base-uncased)",
    )
    p.add_argument(
        "--mlm_checkpoint",
        default=None,
        help="Optional directory with BertForMaskedLM weights; encoder is copied into the classifier body",
    )
    p.add_argument(
        "--pseudo_data",
        default=None,
        type=Path,
        help="Optional CSV/JSONL with text + label (0/1/2) to mix with PhraseBank",
    )
    p.add_argument(
        "--pseudo_weight",
        type=float,
        default=1.0,
        help="Relative sampling weight for pseudo rows vs PhraseBank when merging",
    )
    p.add_argument("--output_dir", required=True, help="Where to save tokenizer + classifier")
    p.add_argument("--max_length", type=int, default=175)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--phrasebank_txt",
        default=None,
        type=Path,
        help="Optional path to Sentences_*.txt; otherwise download PhraseBank next to this file",
    )
    return p.parse_args()


def build_model_and_tokenizer(base_model: str, mlm_checkpoint: str | None):
    """Build the model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)

    # Load the MLM checkpoint if provided
    if mlm_checkpoint:
        mlm = AutoModelForMaskedLM.from_pretrained(mlm_checkpoint)
        if not hasattr(model, "bert"):
            raise ValueError("Expected a BERT-style model with a .bert encoder for --mlm_checkpoint")
        model.bert.load_state_dict(mlm.bert.state_dict())

    return model, tokenizer


def tokenize_dataset(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    """Tokenize the dataset"""
    def _tok(batch):
        """Tokenize the batch"""
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return ds.map(_tok, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    """Compute metrics for the evaluation predictions"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main() -> None:
    """Main function"""
    # Parse the command line arguments
    args = parse_args()
    # Set the random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load the PhraseBank data
    pb_path = args.phrasebank_txt
    # Download the PhraseBank data if not provided
    if pb_path is None:
        pb_path = ensure_finphrasebank()
    df_pb = load_finphrasebank_dataframe(pb_path)

    # Load the pseudo-labeled data if provided
    frames = [df_pb]
    if args.pseudo_data:
        df_p = load_labeled_table(args.pseudo_data)
        if args.pseudo_weight != 1.0:
            n = max(1, int(len(df_p) * args.pseudo_weight))
            df_p = df_p.sample(n=n, random_state=args.seed, replace=len(df_p) < n)
        frames.append(df_p)

    # Concatenate the PhraseBank and pseudo-labeled data
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Build the model and tokenizer
    model, tokenizer = build_model_and_tokenizer(args.base_model, args.mlm_checkpoint)
    # Convert the labels to FinBERT labels if the model uses the FinBERT label order
    if model_uses_finbert_label_order(model):
        df = phrasebank_labels_to_finbert(df)

    # Split the data into training and validation sets
    vc = df["label"].value_counts()
    strat = df["label"] if vc.min() >= 2 else None
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=strat,
    )

    # Convert the data into Hugging Face Datasets
    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))

    # Tokenize the training and validation data
    train_tok = tokenize_dataset(train_ds, tokenizer, args.max_length)
    val_tok = tokenize_dataset(val_ds, tokenizer, args.max_length)

    # Remove the token type ids column
    for col in ("token_type_ids",):
        if col in train_tok.column_names:
            train_tok = train_tok.remove_columns(col)
        if col in val_tok.column_names:
            val_tok = val_tok.remove_columns(col)

    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")

    # Create the data collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create the training arguments
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    if model_uses_finbert_label_order(model):
        permute_classifier_to_phrasebank(model)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved classifier to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
