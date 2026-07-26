"""Collect Lambda: gather raw news/social texts for ONE symbol and write them to S3.

Consumes collect tasks from SQS (batch size 1) and, when anything was written,
enqueues a predict task pointing at the new ``raw/`` object. A direct invoke with
``{"symbol": "AAPL", ...}`` is also accepted for manual replays.

The raw feed is kept durable in S3 so it can be replayed into training and so later
experiments (different models, different thresholds) can reuse the same corpus.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from finsense_shared import normalize_symbol, raw_key
from finsense_shared.aws.s3 import write_jsonl
from finsense_shared.aws.sqs import iter_records, send_json
from finsense_shared.pipeline import TASK_COLLECT, build_predict_task, validate_task
from finsense_shared.sources import collect_for_symbol

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATA_BUCKET = os.environ["DATA_BUCKET"]
PREDICT_QUEUE_URL = os.environ.get("PREDICT_QUEUE_URL", "").strip()
DEFAULT_MAX_ARTICLES = int(os.environ.get("DEFAULT_MAX_ARTICLES", "20"))
INCLUDE_SOCIAL = os.environ.get("INCLUDE_SOCIAL", "true").lower() not in ("0", "false", "no")


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() not in ("0", "false", "no")


def _collect_symbol(symbol: str, run_id: str, max_articles: int, include_social: bool) -> dict[str, Any]:
    """Collect raw items for one ticker and write them to ``raw/dt=.../symbol=.../run_id.jsonl``."""
    items = collect_for_symbol(
        symbol,
        max_articles=max_articles,
        include_social=include_social,
    )
    kept = [it for it in items if (it.text or "").strip()]
    key = raw_key(symbol, run_id)
    now = int(time.time())
    records = [
        {
            "run_id": run_id,
            "symbol": symbol,
            "ingested_at": now,
            "title": it.title,
            "url": it.url,
            "text": it.text,
            "source_type": it.source_type,
        }
        for it in kept
    ]
    written = write_jsonl(DATA_BUCKET, key, records) if records else 0
    return {
        "symbol": symbol,
        "run_id": run_id,
        "bucket": DATA_BUCKET,
        "key": key if written else "",
        "count": written,
        "collected": len(items),
        "dropped_empty": len(items) - len(kept),
    }


def _handle_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Collect one symbol and enqueue the predict task when rows were written."""
    sym = normalize_symbol(str(payload.get("symbol") or ""))
    if not sym:
        return {"error": "invalid_symbol", "payload": payload}
    run_id = str(payload.get("run_id") or "")
    run_id = run_id or f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}"
    opts = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    max_articles = int(opts.get("max_articles", DEFAULT_MAX_ARTICLES))
    max_articles = max(1, min(max_articles, 40))
    include_social = _parse_bool(opts.get("include_social"), INCLUDE_SOCIAL)

    result = _collect_symbol(sym, run_id, max_articles, include_social)

    result["dispatched"] = False
    if result["count"] and PREDICT_QUEUE_URL:
        send_json(
            PREDICT_QUEUE_URL,
            build_predict_task(
                run_id,
                sym,
                bucket=result["bucket"],
                key=result["key"],
                count=result["count"],
            ),
        )
        result["dispatched"] = True

    logger.info("collect_complete %s", {k: result[k] for k in ("symbol", "run_id", "count", "dispatched")})
    return result


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """SQS consumer (batch size 1); also accepts a direct-invoke payload for manual replay."""
    if isinstance(event, dict) and "Records" in event:
        results = []
        for message_id, body in iter_records(event):
            # A body that can't parse will never succeed on retry; surface and drop it.
            if body is None:
                logger.error("invalid_message_body message_id=%s", message_id)
                continue
            results.append(_handle_payload(validate_task(body, TASK_COLLECT)))
        return {"results": results}
    return _handle_payload(event if isinstance(event, dict) else {})
