"""Batch inference for saved sequence-classification checkpoints (aligned with training defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import SENTIMENT_ID_TO_STR

INFERENCE_API_VERSION = "1"

# Default truncation length; must match ``train_classifier`` default unless documented otherwise.
DEFAULT_MAX_LENGTH = 175


@dataclass
class PredictionRecord:
    """One input row and model output (or a structured error)."""

    text: str
    label_id: int | None = None
    label_name: str | None = None
    probabilities: dict[str, float] | None = None
    error: str | None = None
    inference_api_version: str = field(default=INFERENCE_API_VERSION, repr=False)


def _prob_dict(probs_row: np.ndarray) -> dict[str, float]:
    """Map sentiment label IDs to probabilities."""
    return {SENTIMENT_ID_TO_STR[i]: float(probs_row[i]) for i in range(len(probs_row))}


class SentimentPredictor:
    """
    Load a saved Hugging Face sequence-classification directory and run batched inference.

    Tokenization matches training-style defaults: ``truncation=True``, dynamic padding per batch,
    ``max_length`` from training unless overridden at construction time.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str | None = None,
    ) -> None:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 32,
    ) -> list[PredictionRecord]:
        """
        Return one ``PredictionRecord`` per input, in order.

        Empty or whitespace-only strings are not forwarded to the model; they yield
        ``error="empty_or_whitespace_text"`` and null label fields.
        """
        text_list = [t if isinstance(t, str) else str(t) for t in texts]
        out: list[PredictionRecord] = [PredictionRecord(text=text_list[i]) for i in range(len(text_list))]
        pending_idx: list[int] = []
        pending_text: list[str] = []

        for i, raw in enumerate(text_list):
            if not raw.strip():
                out[i] = PredictionRecord(
                    text=raw,
                    label_id=None,
                    label_name=None,
                    probabilities=None,
                    error="empty_or_whitespace_text",
                )
            else:
                pending_idx.append(i)
                pending_text.append(raw)

        for start in range(0, len(pending_text), batch_size):
            chunk = pending_text[start : start + batch_size]
            idx_chunk = pending_idx[start : start + batch_size]
            enc: dict[str, Any] = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = probs.argmax(axis=-1)
            for j, row_i in enumerate(idx_chunk):
                pid = int(pred_ids[j])
                out[row_i] = PredictionRecord(
                    text=text_list[row_i],
                    label_id=pid,
                    label_name=SENTIMENT_ID_TO_STR[pid],
                    probabilities=_prob_dict(probs[j]),
                    error=None,
                )
        return out

    def predict_one(self, text: str, *, batch_size: int = 32) -> PredictionRecord:
        """Convenience wrapper around ``predict`` for a single string."""
        return self.predict([text], batch_size=batch_size)[0]
