"""
Pre-training with masked language modeling on unlabeled financial text.

Feed one or more JSONL files with a "text" field per line, or plain .txt files (one doc per line).

Example:
  python -m training.train_mlm --train_files data/wsb.jsonl data/reuters_lines.txt --output_dir outputs/mlm_bert
  finsense-train-mlm --train_files data/wsb.jsonl --output_dir outputs/mlm_bert
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, Features, Value
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .common import trainer_warmup_steps


def load_texts_from_jsonl(path: Path, text_key: str) -> list[str]:
    """Load texts from a JSONL file"""
    import json

    texts: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get(text_key)
            if t is None:
                continue
            t = str(t).strip()
            if t:
                texts.append(t)
    return texts


def load_texts_from_txt(path: Path) -> list[str]:
    """Load texts from a .txt file"""
    texts: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            t = line.strip()
            if t:
                texts.append(t)
    return texts


def build_dataset(train_files: list[Path], text_key: str) -> Dataset:
    """Build a dataset from the training files"""
    all_texts: list[str] = []
    for p in train_files:
        p = Path(p)
        if p.suffix.lower() == ".jsonl":
            all_texts.extend(load_texts_from_jsonl(p, text_key))
        elif p.suffix.lower() == ".txt":
            all_texts.extend(load_texts_from_txt(p))
        else:
            raise ValueError(f"Unsupported file type {p.suffix}: {p}")

    if not all_texts:
        raise ValueError("No training texts found; check paths and JSONL text key.")

    features = Features({"text": Value("string")})
    ds = Dataset.from_dict({"text": all_texts}, features=features)
    return ds


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    p = argparse.ArgumentParser(description="Financial-domain MLM continued pre-training")
    p.add_argument(
        "--train_files",
        nargs="+",
        required=True,
        help="JSONL (object with text field) and/or .txt (one sentence/doc per line) files",
    )
    p.add_argument("--text_key", default="text", help="JSONL field name for body text")
    p.add_argument(
        "--model_name",
        default="bert-base-uncased",
        help="Base checkpoint (use bert-base-uncased to stay compatible with FinBERT heads)",
    )
    p.add_argument("--output_dir", required=True, help="Directory to save MLM weights + tokenizer")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--fp16", action="store_true", help="Use mixed precision (CUDA only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_paths = [Path(x) for x in args.train_files]
    for p in train_paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    raw_ds = build_dataset(train_paths, args.text_key)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    def tokenize(batch):
        """Tokenize the dataset"""
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized = raw_ds.map(
        tokenize,
        batched=True,
        num_proc=1,
        remove_columns=raw_ds.column_names,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = trainer_warmup_steps(
        num_train_examples=len(tokenized),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=max(
            10,
            len(tokenized) // max(1, args.per_device_train_batch_size * 100),
        ),
        save_strategy="epoch",
        seed=args.seed,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved MLM checkpoint to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
