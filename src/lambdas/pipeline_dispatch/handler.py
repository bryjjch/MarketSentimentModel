"""Dispatch Lambda: enumerate tickers, mint a run_id, enqueue one collect task per symbol.

Invocation modes:
  1. EventBridge daily cron with no payload -> enqueues a collect task for every ticker
     from SSM onto the collect queue.
  2. Direct invoke with ``{"symbol": "AAPL", ...}`` to enqueue a single ticker on demand.

All tasks of one run share the same ``run_id`` so partitions line up for downstream
analytics. This function only enqueues — collection, prediction, labeling, and cache
writes each live in their own queue-driven Lambda.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from botocore.exceptions import ClientError

from finsense_shared import normalize_symbol
from finsense_shared.aws.sqs import send_json
from finsense_shared.pipeline import build_collect_task
from finsense_shared.tickers import load_tickers

logger = logging.getLogger()
logger.setLevel(logging.INFO)

COLLECT_QUEUE_URL = os.environ["COLLECT_QUEUE_URL"]
DEFAULT_MAX_ARTICLES = int(os.environ.get("DEFAULT_MAX_ARTICLES", "20"))
INCLUDE_SOCIAL = os.environ.get("INCLUDE_SOCIAL", "true").lower() not in ("0", "false", "no")


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() not in ("0", "false", "no")


def _dt_date() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    event = event if isinstance(event, dict) else {}
    opts = event.get("options") if isinstance(event.get("options"), dict) else {}
    max_articles = int(opts.get("max_articles", DEFAULT_MAX_ARTICLES))
    max_articles = max(1, min(max_articles, 40))
    include_social = _parse_bool(opts.get("include_social"), INCLUDE_SOCIAL)
    run_id = str(event.get("run_id") or "") or f"{_dt_date()}-{uuid.uuid4().hex[:8]}"

    raw_sym = event.get("symbol")
    if raw_sym is not None:
        sym = normalize_symbol(str(raw_sym))
        if not sym:
            return {"error": "invalid_symbol"}
        symbols = [sym]
    else:
        symbols = list(load_tickers())

    enqueued = 0
    for sym in symbols:
        task = build_collect_task(run_id, sym, max_articles=max_articles, include_social=include_social)
        try:
            send_json(COLLECT_QUEUE_URL, task)
            enqueued += 1
        except ClientError as e:
            logger.exception("collect_enqueue_failed %s: %s", sym, e)

    summary = {"run_id": run_id, "dt": _dt_date(), "symbols": len(symbols), "enqueued": enqueued}
    logger.info("dispatch_complete %s", summary)
    return summary
