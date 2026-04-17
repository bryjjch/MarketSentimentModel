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
    """Map sentiment label IDs to probabilities."""
    return {SENTIMENT_ID_TO_STR[i]: float(probs_row[i]) for i in range(len(probs_row))}


def _record(
    text: str,
    *,
    label_id: int | None,
    label_name: str | None,
    probabilities: dict[str, float] | None,
    error: str | None,
) -> dict[str, Any]:
    """Create a prediction record."""
    return {
        "text": text,
        "label_id": label_id,
        "label_name": label_name,
        "probabilities": probabilities,
        "error": error,
        "inference_api_version": INFERENCE_API_VERSION,
    }


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load the model and tokenizer."""
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # Move the model to the device
    model.to(device)
    # Set the model to evaluation mode
    model.eval()
    # Return the model, tokenizer, and device
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "max_length": DEFAULT_MAX_LENGTH,
    }


def input_fn(request_body: bytes | str, request_content_type: str) -> list[str]:
    """Parse the input request body."""
    # Check if the content type is supported
    if request_content_type and "json" not in request_content_type.lower():
        raise ValueError(f"Unsupported content type: {request_content_type!r} (expected JSON)")
    # Decode the request body
    raw = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    # Load the payload
    payload = json.loads(raw)
    # Check if the payload has texts
    if "texts" in payload:
        # Get the texts
        texts = payload["texts"]
        # Check if the texts are a list
        if not isinstance(texts, list):
            raise ValueError('"texts" must be a JSON array of strings')
        # Return the texts
        return [t if isinstance(t, str) else str(t) for t in texts]
    # Check if the payload has a text
    if "text" in payload:
        # Get the text
        t = payload["text"]
        return [t if isinstance(t, str) else str(t)]
    raise ValueError('JSON body must include "text" (string) or "texts" (array of strings)')


def predict_fn(text_list: list[str], context: dict[str, Any]) -> list[dict[str, Any]]:
    """Predict the sentiment of the text."""
    # Get the model, tokenizer, device, and max length
    model = context["model"]
    tokenizer = context["tokenizer"]
    device: torch.device = context["device"]
    max_length: int = context["max_length"]
    # Set the batch size
    batch_size = 32
    # Create the output list
    out: list[dict[str, Any] | None] = [None] * len(text_list)
    # Create the pending index list
    pending_idx: list[int] = []
    # Create the pending text list
    pending_text: list[str] = []

    # Check each text in the list
    for i, raw in enumerate(text_list):
        # Check if the text is empty or whitespace
        if not raw.strip():
            # Set the error and return the PredictionRecord
            out[i] = _record(
                raw,
                label_id=None,
                label_name=None,
                probabilities=None,
                error="empty_or_whitespace_text",
            )
        else:
            # Add the index and text to the pending lists
            pending_idx.append(i)
            pending_text.append(raw)

    # Process the pending text in batches
    with torch.inference_mode():
        # For each batch
        for start in range(0, len(pending_text), batch_size):
            # Get the chunk of text and index
            chunk = pending_text[start : start + batch_size]
            idx_chunk = pending_idx[start : start + batch_size]
            # Tokenize the text
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # Move the tokens to the device
            enc = {k: v.to(device) for k, v in enc.items()}
            # Get the logits from the model
            logits = model(**enc).logits
            # Get the probabilities from the logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # Get the predicted IDs from the probabilities
            pred_ids = probs.argmax(axis=-1)
            # For each row in the batch, set the PredictionRecord
            for j, row_i in enumerate(idx_chunk):
                # Get the predicted ID
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
    """Output the prediction."""
    # Check if the content type is supported
    if response_content_type and "json" not in response_content_type.lower():
        raise ValueError(f"Unsupported accept type: {response_content_type!r}")
    # Dump the prediction to a JSON string
    return json.dumps(prediction).encode("utf-8")
