"""Predict Lambda: read one raw partition, run SageMaker, split high/low confidence.

Consumes predict tasks from SQS (batch size 1); a direct invoke with the same payload
shape is accepted for manual replays.

Outputs:
  * ``predictions/dt=.../symbol=.../<run_id>.jsonl`` — one row per input text with model
    probabilities + confidence metric.
  * ``curated/dt=.../symbol=.../<run_id>.jsonl`` — **high-confidence** rows only (model
    label trusted) so downstream training has immediate labeled data.
  * A label task on the label queue with the **low-confidence** row indices (pointer
    form — the label Lambda re-reads the rows from the predictions object).
  * A cache-write task on the cache-write queue with the aggregated per-symbol
    sentiment; the cache_write Lambda owns the DynamoDB write.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from botocore.exceptions import ClientError

from finsense_shared import (
    confidence_from_probabilities,
    curated_key,
    dt_from_key,
    is_low_confidence,
    prediction_key,
    recent_headlines,
)
from finsense_shared.aggregate import aggregate_predictions
from finsense_shared.messages import TASK_PREDICT, build_cache_write_task, build_label_task, validate_task
from finsense_shared.s3io import read_jsonl, write_jsonl
from finsense_shared.sagemaker import invoke_predict
from finsense_shared.sqs import iter_records, send_json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
DATA_BUCKET = os.environ["DATA_BUCKET"]
LABEL_QUEUE_URL = os.environ.get("LABEL_QUEUE_URL", "").strip()
CACHE_WRITE_QUEUE_URL = os.environ.get("CACHE_WRITE_QUEUE_URL", "").strip()
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "604800"))
RECENT_HEADLINES_MAX = int(os.environ.get("RECENT_HEADLINES_MAX", "10"))
LOW_CONF_TOP_PROB = float(os.environ.get("LOW_CONF_TOP_PROB", "0.65"))
LOW_CONF_MARGIN = float(os.environ.get("LOW_CONF_MARGIN", "0.0"))
BATCH_SIZE = int(os.environ.get("SAGEMAKER_BATCH_SIZE", "32"))


def _enqueue_cache_write(symbol: str, score: float, label: str, analyzed: int, headlines: list[dict[str, str]]) -> None:
    """Best-effort: a lost cache refresh only delays the UI until the next run."""
    if not CACHE_WRITE_QUEUE_URL:
        return
    try:
        send_json(
            CACHE_WRITE_QUEUE_URL,
            build_cache_write_task(
                symbol,
                score=score,
                label=label,
                article_count=analyzed,
                recent_headlines=headlines,
                updated_at=int(time.time()),
                ttl_seconds=CACHE_TTL_SECONDS,
                source="pipeline",
            ),
        )
    except ClientError as e:
        logger.exception("cache_write_enqueue_failed %s: %s", symbol, e)


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
    low_conf_indices: list[int] = []
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
            low_conf_indices.append(i)
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

    agg_score, agg_label, analyzed = aggregate_predictions(
        [r for r in pred_records if r.get("probabilities") and not r.get("error")]
    )
    headlines = recent_headlines(pred_records, RECENT_HEADLINES_MAX)
    _enqueue_cache_write(symbol, agg_score, agg_label, analyzed, headlines)

    dispatched = False
    if low_conf_indices and LABEL_QUEUE_URL:
        send_json(
            LABEL_QUEUE_URL,
            build_label_task(
                run_id,
                symbol,
                bucket=bucket,
                dt=when.isoformat() if when else None,
                predictions_key=pred_key,
                row_indices=low_conf_indices,
            ),
        )
        dispatched = True

    return {
        "symbol": symbol,
        "run_id": run_id,
        "predictions": len(pred_records),
        "high_confidence": len(curated_hi),
        "low_confidence": len(low_conf_indices),
        "label_dispatched": dispatched,
        "score": round(agg_score, 6),
        "label": agg_label,
        "analyzed": analyzed,
        "predictions_key": pred_key,
        "curated_key": curated_key_hi if curated_hi else "",
    }


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """SQS consumer (batch size 1); also accepts a direct-invoke payload for manual replay."""
    if isinstance(event, dict) and "Records" in event:
        results = []
        for message_id, body in iter_records(event):
            if body is None:
                logger.error("invalid_message_body message_id=%s", message_id)
                continue
            results.append(_predict_for_payload(validate_task(body, TASK_PREDICT)))
        return {"results": results}
    return _predict_for_payload(event if isinstance(event, dict) else {})
