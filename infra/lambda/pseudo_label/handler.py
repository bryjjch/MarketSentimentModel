"""Pseudo-label Lambda: relabel low-confidence prediction rows with a provider-agnostic LLM.

Invocation payload (from the prediction Lambda)::

    {
      "run_id": "...",
      "symbol": "AAPL",
      "bucket": "finsense-data-...",
      "predictions_key": "predictions/dt=.../symbol=AAPL/<run_id>.jsonl",
      "rows": [
        {"row_index": 3, "text": "...", "model_label_id": 1, "probabilities": {...}, "confidence": {...}},
        ...
      ]
    }

Outputs:
  * ``pseudo/dt=.../symbol=.../<run_id>.jsonl`` — one LLM-labeled row per low-confidence
    input (includes both model + LLM labels so downstream code can compare/tune).
  * Appends the same rows (with the LLM label as ground-truth) to
    ``curated/dt=.../symbol=.../<run_id>.jsonl`` by writing a sibling curated object
    suffixed with ``-pseudo`` (so we don't overwrite the high-confidence object already
    emitted by the prediction Lambda).

The provider is resolved from the ``LLM_PROVIDER`` env var (``openai``, ``google``,
``echo``) and the model from ``LLM_MODEL``. API keys come from ``OPENAI_API_KEY`` /
``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` or an AWS Secrets Manager ARN via
``OPENAI_SECRET_ARN`` / ``GOOGLE_SECRET_ARN``.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date
from typing import Any

from finsense_shared import curated_key, dt_from_key, pseudo_label_key
from finsense_shared.llm_label import pseudo_label_text
from finsense_shared.s3io import write_jsonl

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATA_BUCKET = os.environ["DATA_BUCKET"]
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip() or None
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "15"))
LLM_MAX_CHARS = int(os.environ.get("LLM_MAX_CHARS", "4000"))
LLM_SEED = int(os.environ.get("LLM_SEED", "42"))


def _label_row(row: dict[str, Any]) -> dict[str, Any]:
    """Run the LLM for one row; return an enriched record (possibly with ``error``)."""
    text = str(row.get("text") or "")
    try:
        out = pseudo_label_text(
            text,
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            timeout_s=LLM_TIMEOUT_S,
            max_chars=LLM_MAX_CHARS,
            seed=LLM_SEED,
        )
        return {
            "row_index": row.get("row_index"),
            "text": text,
            "title": row.get("title"),
            "url": row.get("url"),
            "source_type": row.get("source_type"),
            "pseudo_label_id": out.label_id,
            "pseudo_label_name": out.label_name,
            "pseudo_provider": out.provider,
            "pseudo_model": out.model,
            "model_label_id": row.get("model_label_id"),
            "model_label_name": row.get("model_label_name"),
            "probabilities": row.get("probabilities"),
            "confidence": row.get("confidence"),
        }
    except Exception as e:  # noqa: BLE001 -- provider failures should not kill the batch
        logger.warning("pseudo_label_failed row=%s err=%s", row.get("row_index"), e)
        return {
            "row_index": row.get("row_index"),
            "text": text,
            "error": str(e),
            "model_label_id": row.get("model_label_id"),
            "model_label_name": row.get("model_label_name"),
            "probabilities": row.get("probabilities"),
        }


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """Label the low-confidence rows and write pseudo/ + curated/ partitions."""
    bucket = event.get("bucket") or DATA_BUCKET
    symbol = str(event.get("symbol") or "").upper()
    run_id = str(event.get("run_id") or "")
    rows = event.get("rows") or []

    if not symbol or not run_id:
        return {"error": "missing_payload_fields"}
    if not isinstance(rows, list) or not rows:
        return {"symbol": symbol, "run_id": run_id, "labeled": 0, "detail": "no_rows"}

    # Derive the partition date so pseudo/ and curated/ land under the same dt=
    # partition as the predictions that triggered this invocation.
    when: date | None = None
    dt_str = str(event.get("dt") or "")
    if dt_str:
        try:
            when = date.fromisoformat(dt_str)
        except ValueError:
            pass
    if when is None:
        when = dt_from_key(str(event.get("predictions_key") or ""))

    now = int(time.time())
    labeled: list[dict[str, Any]] = []
    curated_rows: list[dict[str, Any]] = []

    for r in rows:
        if not isinstance(r, dict):
            continue
        enriched = _label_row(r)
        enriched["pseudo_labeled_at"] = now
        enriched["run_id"] = run_id
        enriched["symbol"] = symbol
        labeled.append(enriched)

        if "pseudo_label_id" in enriched and enriched.get("error") is None:
            curated_rows.append({
                "run_id": run_id,
                "symbol": symbol,
                "text": enriched.get("text"),
                "label_id": enriched["pseudo_label_id"],
                "label_name": enriched["pseudo_label_name"],
                "probabilities": enriched.get("probabilities"),
                "confidence": enriched.get("confidence"),
                "source": "pseudo",
                "pseudo_provider": enriched["pseudo_provider"],
                "pseudo_model": enriched["pseudo_model"],
                "model_label_id": enriched.get("model_label_id"),
                "model_label_name": enriched.get("model_label_name"),
                "created_at": now,
            })

    pseudo_key = pseudo_label_key(symbol, run_id, when=when)
    write_jsonl(bucket, pseudo_key, labeled)

    curated_out = ""
    if curated_rows:
        curated_out = curated_key(symbol, f"{run_id}-pseudo", when=when)
        write_jsonl(bucket, curated_out, curated_rows)

    n_ok = sum(1 for r in labeled if "pseudo_label_id" in r and r.get("error") is None)
    summary = {
        "symbol": symbol,
        "run_id": run_id,
        "labeled": n_ok,
        "failed": len(labeled) - n_ok,
        "pseudo_key": pseudo_key,
        "curated_key": curated_out,
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL or "default",
    }
    logger.info("pseudo_label_complete %s", summary)
    return summary
