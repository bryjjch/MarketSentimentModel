"""
SageMaker Hugging Face inference entrypoint.

Mirrors ``training.inference.SentimentPredictor`` (max_length=175, empty/whitespace handling,
label IDs 0/1/2) without importing the training package (this file ships inside ``model.tar.gz``).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

INFERENCE_API_VERSION = "1"
DEFAULT_MAX_LENGTH = 175
SENTIMENT_ID_TO_STR = {0: "negative", 1: "neutral", 2: "positive"}


def _prob_dict(probs_row: np.ndarray) -> dict[str, float]:
    """Map label indices to human-readable probability keys."""
    return {SENTIMENT_ID_TO_STR[i]: float(probs_row[i]) for i in range(len(probs_row))}


def _record(
    text: str,
    *,
    label_id: int | None,
    label_name: str | None,
    probabilities: dict[str, float] | None,
    error: str | None,
) -> dict[str, Any]:
    """One JSON object per input text (matches training ``PredictionRecord`` fields)."""
    return {
        "text": text,
        "label_id": label_id,
        "label_name": label_name,
        "probabilities": probabilities,
        "error": error,
        "inference_api_version": INFERENCE_API_VERSION,
    }


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load tokenizer and model; return a context dict for ``predict_fn``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "max_length": DEFAULT_MAX_LENGTH,
    }


def input_fn(request_body: bytes | str, request_content_type: str) -> list[str]:
    """Parse JSON body: either ``{"texts": [...]}`` or ``{"text": "..."}``."""
    if request_content_type and "json" not in request_content_type.lower():
        raise ValueError(f"Unsupported content type: {request_content_type!r} (expected JSON)")
    raw = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    payload = json.loads(raw)
    if "texts" in payload:
        texts = payload["texts"]
        if not isinstance(texts, list):
            raise ValueError('"texts" must be a JSON array of strings')
        return [t if isinstance(t, str) else str(t) for t in texts]
    if "text" in payload:
        t = payload["text"]
        return [t if isinstance(t, str) else str(t)]
    raise ValueError('JSON body must include "text" (string) or "texts" (array of strings)')


def predict_fn(text_list: list[str], context: dict[str, Any]) -> list[dict[str, Any]]:
    """Batch inference with empty-string short-circuit (same semantics as ``SentimentPredictor``)."""
    model = context["model"]
    tokenizer = context["tokenizer"]
    device: torch.device = context["device"]
    max_length: int = context["max_length"]
    batch_size = 32
    out: list[dict[str, Any] | None] = [None] * len(text_list)
    pending_idx: list[int] = []
    pending_text: list[str] = []

    for i, raw in enumerate(text_list):
        if not raw.strip():
            out[i] = _record(
                raw,
                label_id=None,
                label_name=None,
                probabilities=None,
                error="empty_or_whitespace_text",
            )
        else:
            pending_idx.append(i)
            pending_text.append(raw)

    with torch.inference_mode():
        for start in range(0, len(pending_text), batch_size):
            chunk = pending_text[start : start + batch_size]
            idx_chunk = pending_idx[start : start + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = probs.argmax(axis=-1)
            for j, row_i in enumerate(idx_chunk):
                pid = int(pred_ids[j])
                out[row_i] = _record(
                    text_list[row_i],
                    label_id=pid,
                    label_name=SENTIMENT_ID_TO_STR[pid],
                    probabilities=_prob_dict(probs[j]),
                    error=None,
                )

    if any(r is None for r in out):
        raise RuntimeError("internal error: unfilled prediction slots")
    return out  # type: ignore[return-value]


def output_fn(prediction: list[dict[str, Any]], response_content_type: str) -> bytes:
    """Serialize predictions as UTF-8 JSON bytes."""
    if response_content_type and "json" not in response_content_type.lower():
        raise ValueError(f"Unsupported accept type: {response_content_type!r}")
    return json.dumps(prediction).encode("utf-8")
