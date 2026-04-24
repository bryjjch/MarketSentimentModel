"""Daily ingestion Lambda: for each ticker, collect raw news/social texts and write them to S3.

Invocation modes:
  1. EventBridge daily cron with no payload -> enumerates tickers from SSM and fans out to
     the prediction Lambda with a per-symbol S3 pointer.
  2. Direct invoke with ``{"symbol": "AAPL", ...}`` to ingest a single ticker on demand.

We intentionally separate ingestion from prediction so the raw feed is durable (can be
replayed into training) and so later experiments (different models, different
thresholds) can reuse the same corpus. Each daily fan-out uses the same ``run_id``
across all three Lambdas so partitions line up for downstream analytics.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from finsense_shared import normalize_symbol, partition_prefix, raw_key
from finsense_shared.s3io import write_jsonl
from finsense_shared.sources import collect_for_symbol
from finsense_shared.tickers import load_tickers

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATA_BUCKET = os.environ["DATA_BUCKET"]
PREDICTION_FUNCTION_NAME = os.environ.get("PREDICTION_FUNCTION_NAME", "").strip()
DEFAULT_MAX_ARTICLES = int(os.environ.get("DEFAULT_MAX_ARTICLES", "20"))
INCLUDE_SOCIAL = os.environ.get("INCLUDE_SOCIAL", "true").lower() not in ("0", "false", "no")

_lambda = boto3.client("lambda")


def _parse_bool(value: Any, default: bool) -> bool:
    """Return a bool from *value*, applying the same rules as the env-var parsing above.

    Accepts actual JSON booleans unchanged.  String values are treated case-insensitively:
    ``"0"``, ``"false"``, and ``"no"`` are falsy; everything else is truthy.
    Falls back to *default* when *value* is ``None``.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() not in ("0", "false", "no")


def _now_ts() -> int:
    return int(time.time())


def _dt_date() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _ingest_symbol(symbol: str, run_id: str, max_articles: int, include_social: bool) -> dict[str, Any]:
    """Collect raw items for one ticker and write them to ``raw/dt=.../symbol=.../run_id.jsonl``."""
    items = collect_for_symbol(
        symbol,
        max_articles=max_articles,
        include_social=include_social,
    )
    kept = [it for it in items if (it.text or "").strip()]
    key = raw_key(symbol, run_id)
    now = _now_ts()
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


def _fan_out_to_prediction(payloads: list[dict[str, Any]]) -> dict[str, int]:
    """Async-invoke the prediction Lambda per ticker. Returns ok/failed counts."""
    if not PREDICTION_FUNCTION_NAME:
        return {"dispatched": 0, "failed": 0}
    ok = 0
    failed = 0
    for p in payloads:
        if not p.get("count"):
            continue
        try:
            _lambda.invoke(
                FunctionName=PREDICTION_FUNCTION_NAME,
                InvocationType="Event",
                Payload=json.dumps(p).encode("utf-8"),
            )
            ok += 1
        except ClientError as e:
            logger.exception("prediction_dispatch_failed %s: %s", p.get("symbol"), e)
            failed += 1
    return {"dispatched": ok, "failed": failed}


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """Entrypoint. Handles EventBridge cron fan-out and direct single-symbol invokes."""
    raw_sym = (event or {}).get("symbol")
    opts = event.get("options") if isinstance(event, dict) and isinstance(event.get("options"), dict) else {}
    max_articles = int(opts.get("max_articles", DEFAULT_MAX_ARTICLES))
    max_articles = max(1, min(max_articles, 40))
    include_social = _parse_bool(opts.get("include_social"), INCLUDE_SOCIAL)
    run_id = str(event.get("run_id") or "") if isinstance(event, dict) else ""
    run_id = run_id or f"{_dt_date()}-{uuid.uuid4().hex[:8]}"

    if raw_sym is not None:
        sym = normalize_symbol(str(raw_sym))
        if not sym:
            return {"error": "invalid_symbol"}
        result = _ingest_symbol(sym, run_id, max_articles, include_social)
        dispatch = _fan_out_to_prediction([result])
        return {"run_id": run_id, "symbols": 1, **dispatch, "results": [result]}

    tickers = load_tickers()
    results: list[dict[str, Any]] = []
    for sym in tickers:
        try:
            results.append(_ingest_symbol(sym, run_id, max_articles, include_social))
        except Exception as e:  # noqa: BLE001 -- one bad symbol shouldn't kill the batch
            logger.exception("ingest_failed %s: %s", sym, e)
            results.append({"symbol": sym, "run_id": run_id, "count": 0, "error": str(e)})

    dispatch = _fan_out_to_prediction(results)
    summary = {
        "run_id": run_id,
        "dt": _dt_date(),
        "symbols": len(tickers),
        "ingested": sum(r.get("count", 0) for r in results),
        "raw_prefix": partition_prefix("raw", symbol="", when=None).replace("symbol=/", ""),
        **dispatch,
    }
    logger.info("ingest_complete %s", summary)
    summary["results"] = results
    return summary
