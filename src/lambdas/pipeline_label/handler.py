"""Label Lambda: relabel low-confidence prediction rows with a provider-agnostic LLM.

Consumes label tasks from SQS (batch size 1). The canonical payload is the pointer
form emitted by the predict Lambda::

    {
      "task": "label",
      "run_id": "...",
      "symbol": "AAPL",
      "bucket": "finsense-data-...",
      "dt": "2026-07-25",
      "predictions_key": "predictions/dt=.../symbol=AAPL/<run_id>.jsonl",
      "row_indices": [1, 3]
    }

The referenced rows are re-read from the predictions object by ``row_index`` so the
message stays tiny regardless of text size. A direct invoke with inline ``rows``
(the pre-SQS payload shape) is still accepted for manual replays.

Outputs:
  * ``pseudo/dt=.../symbol=.../<run_id>.jsonl`` — one LLM-labeled row per low-confidence
    input (includes both model + LLM labels so downstream code can compare/tune).
  * ``curated/dt=.../symbol=.../<run_id>-pseudo.jsonl`` — the successfully labeled rows
    with the LLM label as ground truth (sibling of the high-confidence curated object,
    which the predict Lambda already wrote).

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
from finsense_shared.aws.s3 import read_jsonl, write_jsonl
from finsense_shared.aws.sqs import iter_records
from finsense_shared.llm_label import pseudo_label_text
from finsense_shared.pipeline import TASK_LABEL, validate_task

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


def _rows_from_pointer(bucket: str, predictions_key: str, row_indices: list[Any]) -> list[dict[str, Any]]:
    """Select the low-confidence rows out of the predictions object by ``row_index``."""
    want = {int(i) for i in row_indices}
    out: list[dict[str, Any]] = []
    for rec in read_jsonl(bucket, predictions_key):
        try:
            idx = int(rec.get("row_index", -1))
        except (TypeError, ValueError):
            continue
        if idx not in want:
            continue
        out.append({
            "row_index": idx,
            "text": rec.get("text"),
            "title": rec.get("title"),
            "url": rec.get("url"),
            "source_type": rec.get("source_type"),
            "model_label_id": rec.get("label_id"),
            "model_label_name": rec.get("label_name"),
            "probabilities": rec.get("probabilities"),
            "confidence": rec.get("confidence"),
        })
    return out


def _label_payload(event: dict[str, Any]) -> dict[str, Any]:
    """Label the low-confidence rows and write pseudo/ + curated/ partitions."""
    bucket = event.get("bucket") or DATA_BUCKET
    symbol = str(event.get("symbol") or "").upper()
    run_id = str(event.get("run_id") or "")

    if not symbol or not run_id:
        return {"error": "missing_payload_fields"}

    rows = event.get("rows")
    if not isinstance(rows, list) or not rows:
        row_indices = event.get("row_indices")
        predictions_key = str(event.get("predictions_key") or "")
        if isinstance(row_indices, list) and row_indices and predictions_key:
            rows = _rows_from_pointer(bucket, predictions_key, row_indices)
        else:
            rows = []
    if not rows:
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


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """SQS consumer (batch size 1); also accepts a direct-invoke payload for manual replay."""
    if isinstance(event, dict) and "Records" in event:
        results = []
        for message_id, body in iter_records(event):
            if body is None:
                logger.error("invalid_message_body message_id=%s", message_id)
                continue
            results.append(_label_payload(validate_task(body, TASK_LABEL)))
        return {"results": results}
    return _label_payload(event if isinstance(event, dict) else {})
