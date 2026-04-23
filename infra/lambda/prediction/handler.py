"""Prediction Lambda: read one raw partition, run SageMaker, split high/low confidence.

Invocation payload (from the ingestion Lambda)::

    {
      "run_id": "2026-04-23-abcd1234",
      "symbol": "AAPL",
      "bucket": "finsense-data-...",
      "key": "raw/dt=2026-04-23/symbol=AAPL/<run_id>.jsonl",
      "count": 12
    }

Outputs:
  * ``predictions/dt=.../symbol=.../<run_id>.jsonl`` — one row per input text with model
    probabilities + confidence metric.
  * ``curated/dt=.../symbol=.../<run_id>.jsonl`` — **high-confidence** rows only (model
    label trusted) so downstream training has immediate labeled data.
  * Async fan-out to the pseudo-label Lambda with the list of **low-confidence** row
    indices. That Lambda will emit ``pseudo/`` + merge into ``curated/`` with LLM labels.

DynamoDB write: this Lambda also refreshes the per-symbol ``sentiment_cache`` row so the
existing ``GET /sentiment/cache/{symbol}`` API keeps serving data without any extra
Lambda. The old scheduled sentiment_refresh Lambda (which did
``collect -> predict -> PutItem`` once per ticker) has been removed; this function
does the same work as part of the daily ingestion fan-out.
"""

from __future__ import annotations

import json
import logging
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import boto3
from botocore.exceptions import ClientError

from finsense_shared import (
    aggregate_predictions,
    confidence_from_probabilities,
    curated_key,
    dt_from_key,
    is_low_confidence,
    prediction_key,
)
from finsense_shared.s3io import read_jsonl, write_jsonl
from finsense_shared.sagemaker import invoke_predict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
DATA_BUCKET = os.environ["DATA_BUCKET"]
PSEUDO_LABEL_FUNCTION_NAME = os.environ.get("PSEUDO_LABEL_FUNCTION_NAME", "").strip()
CACHE_TABLE_NAME = os.environ.get("CACHE_TABLE_NAME", "").strip()
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "604800"))
RECENT_HEADLINES_MAX = int(os.environ.get("RECENT_HEADLINES_MAX", "10"))
LOW_CONF_TOP_PROB = float(os.environ.get("LOW_CONF_TOP_PROB", "0.65"))
LOW_CONF_MARGIN = float(os.environ.get("LOW_CONF_MARGIN", "0.0"))
BATCH_SIZE = int(os.environ.get("SAGEMAKER_BATCH_SIZE", "32"))

_lambda = boto3.client("lambda")
_ddb = boto3.resource("dynamodb") if CACHE_TABLE_NAME else None


def _to_ddb_number(value: Any, default: str = "0") -> Decimal:
    try:
        d = Decimal(str(value))
        if d.is_nan() or d.is_infinite():
            return Decimal(default)
        return d
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _write_cache_row(symbol: str, score: float, label: str, analyzed: int, headlines: list[dict[str, str]]) -> None:
    """Mirror per-symbol results into DynamoDB so the existing cache_read API keeps working."""
    if not CACHE_TABLE_NAME or _ddb is None:
        return
    try:
        table = _ddb.Table(CACHE_TABLE_NAME)
        now = int(time.time())
        table.put_item(
            Item={
                "symbol": symbol,
                "score": _to_ddb_number(score),
                "label": label,
                "article_count": int(analyzed),
                "recent_headlines": headlines,
                "updated_at": now,
                "expires_at": now + CACHE_TTL_SECONDS,
            }
        )
    except ClientError as e:
        logger.exception("ddb_put_failed %s: %s", symbol, e)


def _dispatch_pseudo_label(payload: dict[str, Any]) -> bool:
    """Fire-and-forget async invoke of the pseudo-label Lambda."""
    if not PSEUDO_LABEL_FUNCTION_NAME:
        return False
    try:
        _lambda.invoke(
            FunctionName=PSEUDO_LABEL_FUNCTION_NAME,
            InvocationType="Event",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        return True
    except ClientError as e:
        logger.exception("pseudo_label_dispatch_failed symbol=%s: %s", payload.get("symbol"), e)
        return False


def _predict_for_payload(event: dict[str, Any]) -> dict[str, Any]:
    """Core prediction + partitioning logic for one raw S3 object."""
    bucket = event.get("bucket") or DATA_BUCKET
    key = event.get("key") or ""
    symbol = (event.get("symbol") or "").upper()
    run_id = event.get("run_id") or ""

    if not key or not symbol or not run_id:
        return {"error": "missing_payload_fields", "event": event}

    # Derive the partition date from the raw key so all output partitions land
    # under the same dt= prefix even on retries or cross-day replays.
    when = dt_from_key(key)

    rows: list[dict[str, Any]] = list(read_jsonl(bucket, key))
    if not rows:
        return {"symbol": symbol, "run_id": run_id, "predictions": 0, "detail": "no_rows"}

    texts = [str(r.get("text") or "") for r in rows]
    records = invoke_predict(ENDPOINT_NAME, texts, batch_size=BATCH_SIZE)

    pred_records: list[dict[str, Any]] = []
    curated_hi: list[dict[str, Any]] = []
    low_conf_rows: list[dict[str, Any]] = []
    now = int(time.time())

    for i, (raw_row, rec) in enumerate(zip(rows, records)):
        probs = rec.get("probabilities") if isinstance(rec, dict) else None
        confidence = confidence_from_probabilities(probs or {}).as_dict() if probs else {}
        enriched: dict[str, Any] = {
            "run_id": run_id,
            "symbol": symbol,
            "text": raw_row.get("text"),
            "title": raw_row.get("title"),
            "url": raw_row.get("url"),
            "source_type": raw_row.get("source_type"),
            "predicted_at": now,
            "label_id": rec.get("label_id"),
            "label_name": rec.get("label_name"),
            "probabilities": probs,
            "confidence": confidence,
            "error": rec.get("error"),
            "row_index": i,
        }
        pred_records.append(enriched)

        if rec.get("error") or not probs:
            continue

        if is_low_confidence(
            probs,
            min_top_prob=LOW_CONF_TOP_PROB,
            min_margin=LOW_CONF_MARGIN,
        ):
            low_conf_rows.append({
                "row_index": i,
                "text": raw_row.get("text"),
                "title": raw_row.get("title"),
                "url": raw_row.get("url"),
                "source_type": raw_row.get("source_type"),
                "model_label_id": rec.get("label_id"),
                "model_label_name": rec.get("label_name"),
                "probabilities": probs,
                "confidence": confidence,
            })
        else:
            curated_hi.append({
                "run_id": run_id,
                "symbol": symbol,
                "text": raw_row.get("text"),
                "label_id": rec.get("label_id"),
                "label_name": rec.get("label_name"),
                "probabilities": probs,
                "confidence": confidence,
                "source": "model",
                "created_at": now,
            })

    pred_key = prediction_key(symbol, run_id, when=when)
    write_jsonl(bucket, pred_key, pred_records)

    curated_key_hi = curated_key(symbol, run_id, when=when)
    if curated_hi:
        write_jsonl(bucket, curated_key_hi, curated_hi)

    # Aggregate + cache row for the UI
    agg_score, agg_label, analyzed = aggregate_predictions(
        [r for r in pred_records if r.get("probabilities") and not r.get("error")]
    )
    headlines = [
        {"title": r.get("title") or "", "url": r.get("url") or ""}
        for r in pred_records[:RECENT_HEADLINES_MAX]
    ]
    _write_cache_row(symbol, agg_score, agg_label, analyzed, headlines)

    dispatched = False
    if low_conf_rows:
        dispatched = _dispatch_pseudo_label({
            "run_id": run_id,
            "symbol": symbol,
            "bucket": bucket,
            "dt": when.isoformat() if when else None,
            "predictions_key": pred_key,
            "rows": low_conf_rows,
        })

    return {
        "symbol": symbol,
        "run_id": run_id,
        "predictions": len(pred_records),
        "high_confidence": len(curated_hi),
        "low_confidence": len(low_conf_rows),
        "pseudo_dispatched": dispatched,
        "score": round(agg_score, 6),
        "label": agg_label,
        "analyzed": analyzed,
        "predictions_key": pred_key,
        "curated_key": curated_key_hi if curated_hi else "",
    }


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """Handle direct invoke from ingestion (single symbol) or batch of events (for manual replay)."""
    if isinstance(event, dict) and "records" in event and isinstance(event["records"], list):
        return {"results": [_predict_for_payload(p) for p in event["records"] if isinstance(p, dict)]}
    return _predict_for_payload(event if isinstance(event, dict) else {})
