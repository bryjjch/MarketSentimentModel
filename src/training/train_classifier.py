"""
Fine-tune a BERT-family classifier on Financial PhraseBank (and optional pseudo-labeled data).

Supports:
- FinBERT or other HF checkpoints (--base_model ProsusAI/finbert)
- Optional encoder weights from MLM continued pre-training (--mlm_checkpoint)
- Merging pseudo-labeled JSONL/CSV (--pseudo_data); train/val/test splits are drawn from
  PhraseBank only, then pseudo rows are added to the training pool only

Saves a Hugging Face model directory plus ``training_manifest.json``.
Use ``training.inference.SentimentPredictor`` for aligned train/serve tokenization.

Example:
  python -m training.train_classifier --base_model ProsusAI/finbert --output_dir outputs/clf_finbert
  python -m training.train_classifier --base_model bert-base-uncased --mlm_checkpoint outputs/mlm_bert --pseudo_data data/pseudo.jsonl --output_dir outputs/clf_mlm
  finsense-train-classifier --base_model ProsusAI/finbert --output_dir outputs/clf_finbert
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .artifacts import args_namespace_to_dict, build_training_manifest, write_training_manifest
from .common import (
    ensure_finphrasebank,
    load_finphrasebank_dataframe,
    load_labeled_table,
    trainer_warmup_steps,
)
from .evaluation import classification_metrics, confusion_matrix_list, metrics_from_eval_pred
from .inference import DEFAULT_MAX_LENGTH

# Financial PhraseBank / deployment label names.
PHRASEBANK_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# ProsusAI FinBERT uses 0=positive, 1=negative, 2=neutral.
PHRASEBANK_TO_FINBERT_LABEL = {0: 1, 1: 2, 2: 0}
_FINBERT_ROW_PERMUTE = [1, 2, 0]  # phrasebank row i <- finbert row perm[i]


def split_labeled_frame(
    df: pd.DataFrame,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    label_column: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Stratified splits when each class has at least two rows in the parent frame.

    If ``test_ratio`` is 0, returns ``(train_df, val_df, None)`` using ``val_ratio``
    as the validation fraction of the full frame (legacy behavior).
    If ``test_ratio`` > 0, first hold out a test fraction, then split the remainder
    into train/val with ``val_ratio`` as the validation fraction of that remainder.
    """
    # Stratify the data by the label column
    strat = None
    if label_column in df.columns:
        vc = df[label_column].value_counts()
        strat = df[label_column] if vc.min() >= 2 else None

    # If test_ratio is greater than 0, split the data into train/val/test
    if test_ratio and test_ratio > 0:
        # Split the data into train/val/test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=seed,
            stratify=strat,
        )
        # Stratify the data by the label column
        strat2 = None
        if label_column in train_val_df.columns:
            vc2 = train_val_df[label_column].value_counts()
            strat2 = train_val_df[label_column] if vc2.min() >= 2 else None
        # Split the data into train/val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=seed,
            stratify=strat2,
        )
        return train_df, val_df, test_df

    # Split the data into train/val
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=strat,
    )
    return train_df, val_df, None


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (optional ``argv`` for tests)."""
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
    p.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Sequence truncation (default {DEFAULT_MAX_LENGTH}; match inference serving)",
    )
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation fraction of the training pool (after an optional test holdout)",
    )
    p.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Hold out this fraction as a frozen test / production-eval set (0 disables)",
    )
    p.add_argument(
        "--metric_for_best_model",
        choices=("accuracy", "macro_f1", "weighted_f1"),
        default="macro_f1",
        help="Metric from compute_metrics used with load_best_model_at_end",
    )
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--phrasebank_txt",
        default=None,
        type=Path,
        help="Optional path to Sentences_*.txt; otherwise download PhraseBank next to this file",
    )
    return p.parse_args(argv)


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


def main() -> None:
    """Main function"""
    # Parse the arguments
    args = parse_args(None)
    # Set the seed for the random number generators
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get the path to the PhraseBank text file
    pb_path = args.phrasebank_txt
    # If the path is not provided, download the PhraseBank text file
    if pb_path is None:
        pb_path = ensure_finphrasebank()
    df_pb = load_finphrasebank_dataframe(pb_path)

    df_p: pd.DataFrame | None = None
    # If pseudo_data is provided, load the labeled table
    if args.pseudo_data:
        # Load the labeled table
        df_p = load_labeled_table(args.pseudo_data)
        # If the pseudo_weight is not 1.0, sample the labeled table
        if args.pseudo_weight != 1.0:
            # Calculate the number of rows to sample
            n = max(1, int(len(df_p) * args.pseudo_weight))
            # Sample the labeled table
            df_p = df_p.sample(n=n, random_state=args.seed, replace=len(df_p) < n)

    model, tokenizer = build_model_and_tokenizer(args.base_model, args.mlm_checkpoint)

    df_pb_split = df_pb.copy()
    if model_uses_finbert_label_order(model):
        df_pb_split = phrasebank_labels_to_finbert(df_pb_split)

    # Split the data into train/val/test
    train_pb, val_df, test_df = split_labeled_frame(
        df_pb_split,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # If pseudo_data is provided, add the pseudo rows to the training pool
    if df_p is not None:
        # Copy the pseudo data
        df_p_train = df_p.copy()
        # If the model uses the FinBERT label order, convert the labels to FinBERT labels
        if model_uses_finbert_label_order(model):
            df_p_train = phrasebank_labels_to_finbert(df_p_train)
        train_df = pd.concat([train_pb, df_p_train], ignore_index=True)
        train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        train_df = train_pb

    # Create the datasets
    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))
    # Tokenize the datasets
    train_tok = tokenize_dataset(train_ds, tokenizer, args.max_length)
    val_tok = tokenize_dataset(val_ds, tokenizer, args.max_length)

    # If test_df is not None, tokenize the test dataset
    test_tok = None
    if test_df is not None:
        test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))
        test_tok = tokenize_dataset(test_ds, tokenizer, args.max_length)

    # Remove the token type IDs column from the datasets
    for col in ("token_type_ids",):
        if col in train_tok.column_names:
            train_tok = train_tok.remove_columns(col)
        if col in val_tok.column_names:
            val_tok = val_tok.remove_columns(col)
        if test_tok is not None and col in test_tok.column_names:
            test_tok = test_tok.remove_columns(col)

    # Rename the label column to labels
    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")
    if test_tok is not None:
        test_tok = test_tok.rename_column("label", "labels")

    # Create the data collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create the output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calculate the warmup steps
    warmup_steps = trainer_warmup_steps(
        num_train_examples=len(train_tok),
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
    )

    # Create the training arguments
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        fp16=args.fp16,
        report_to="none",
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=metrics_from_eval_pred,
    )

    # Train the model
    trainer.train()
    # If the model uses the FinBERT label order, permute the classifier to PhraseBank labels
    if model_uses_finbert_label_order(model):
        permute_classifier_to_phrasebank(model)

    # Evaluate the model
    eval_run = trainer.evaluate(val_tok)
    # Create the evaluation metrics
    eval_metrics = {
        k: float(v)
        for k, v in eval_run.items()
        if isinstance(v, (int, float)) and "runtime" not in k.lower()
    }

    # Predict the labels for the validation dataset
    val_pred = trainer.predict(val_tok)
    # Create the validation labels
    val_labels = np.asarray(val_pred.label_ids).astype(int).ravel()
    # Create the validation logits
    val_logits = val_pred.predictions
    # Create the validation predictions
    val_preds = np.argmax(val_logits, axis=-1)
    # Create the validation confusion matrix
    val_cm = confusion_matrix_list(val_labels, val_preds)

    # Create the test metrics
    test_metrics_flat: dict[str, float] = {}
    # Create the test confusion matrix
    test_cm = None
    if test_tok is not None:
        # Predict the labels for the test dataset
        test_pred = trainer.predict(test_tok)
        # Create the test labels
        t_labels = np.asarray(test_pred.label_ids).astype(int).ravel()
        # Create the test logits
        t_logits = test_pred.predictions
        # Create the test predictions
        t_p = np.argmax(t_logits, axis=-1)
        # Create the test metrics
        test_metrics_flat = {
            f"test_{k}": float(v) for k, v in classification_metrics(t_labels, t_p).items()
        }
        # Create the test confusion matrix
        test_cm = confusion_matrix_list(t_labels, t_p)

    # Save the model
    trainer.save_model(str(out_dir))
    # Save the tokenizer
    tokenizer.save_pretrained(str(out_dir))
    # Print the saved classifier
    print(f"Saved classifier to {out_dir.resolve()}")

    # Create the data summary
    data_summary = {
        "phrasebank_rows": int(len(df_pb)),
        "pseudo_rows": int(len(df_p)) if df_p is not None else 0,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)) if test_df is not None else 0,
    }
    # Get the best validation metric
    best_val = trainer.state.best_metric
    # Build the training manifest
    manifest = build_training_manifest(
        run_configuration=args_namespace_to_dict(args),
        validation_metrics=eval_metrics,
        test_metrics=test_metrics_flat,
        validation_confusion_matrix=val_cm,
        test_confusion_matrix=test_cm,
        best_model_checkpoint=trainer.state.best_model_checkpoint,
        best_metric_name=f"eval_{args.metric_for_best_model}",
        best_metric_value=float(best_val) if best_val is not None else None,
        data_summary=data_summary,
    )
    # Write the training manifest
    manifest_path = write_training_manifest(out_dir, manifest)
    # Print the path to the training manifest
    print(f"Wrote run manifest to {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
