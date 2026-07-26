"""HTTP API (v2) handler: POST /sentiment/by-symbol (symbol to sources to SageMaker to aggregate).

Returns the computed sentiment synchronously in the response body. The only side
effect is a best-effort cache-write task on the cache-write queue — the cache_write
Lambda owns the DynamoDB write, this function has no table access.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from botocore.exceptions import ClientError

from finsense_shared import (
    aggregate_predictions,
    load_valid_ticker_set,
    normalize_symbol,
    recent_headlines,
)
from finsense_shared.http import parse_json_body, response
from finsense_shared.messages import build_cache_write_task
from finsense_shared.sagemaker import invoke_predict
from finsense_shared.sources import collect_for_symbol
from finsense_shared.sources.base import CollectedItem
from finsense_shared.sqs import send_json

ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
RECENT_HEADLINES_MAX = int(os.environ.get("RECENT_HEADLINES_MAX", "10"))
DEFAULT_MAX_ARTICLES = int(os.environ.get("DEFAULT_MAX_ARTICLES", "12"))
CACHE_WRITE_QUEUE_URL = os.environ.get("CACHE_WRITE_QUEUE_URL", "").strip()
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))

logger = logging.getLogger()

_5XX_ERRORS = ("ModelError", "InternalFailure", "InvocationError", "ServiceUnavailable")


def _is_known_symbol(symbol: str) -> bool:
    return symbol in load_valid_ticker_set()


def _enqueue_cache_write(out: dict[str, Any]) -> None:
    """Best-effort cache write so long-tail symbols are reused by cache reads."""
    if not CACHE_WRITE_QUEUE_URL:
        return
    try:
        send_json(
            CACHE_WRITE_QUEUE_URL,
            build_cache_write_task(
                out["symbol"],
                score=out["score"],
                label=out["label"],
                article_count=out["article_count"],
                recent_headlines=out["recent_headlines"],
                updated_at=out["updated_at"],
                ttl_seconds=CACHE_TTL_SECONDS,
                source="api",
            ),
        )
    except ClientError as e:
        logger.exception("cache_write_enqueue_failed %s: %s", out.get("symbol"), e)


def _empty_result(symbol: str, detail: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "score": 0.0,
        "label": "neutral",
        "article_count": 0,
        "recent_headlines": [],
        "sources": {"news_rss": 0, "reddit": 0},
        "updated_at": int(time.time()),
        "detail": detail,
    }


def run_sentiment(symbol: str, options: dict[str, Any] | None) -> dict[str, Any]:
    """Core flow: collect texts, invoke SageMaker, aggregate. Used by API and direct invoke."""
    opts = options or {}
    max_articles = int(opts.get("max_articles", DEFAULT_MAX_ARTICLES))
    max_articles = max(1, min(max_articles, 40))
    include_social = bool(opts.get("include_social", True))

    items: list[CollectedItem] = collect_for_symbol(
        symbol,
        max_articles=max_articles,
        include_social=include_social,
    )
    kept = [it for it in items if it.text.strip()]
    if not kept:
        return _empty_result(symbol, "no_articles_collected" if not items else "no_non_empty_text")

    records = invoke_predict(ENDPOINT_NAME, [it.text for it in kept])
    ok_records = [r for r in records if isinstance(r, dict) and r.get("probabilities") and not r.get("error")]
    if not ok_records:
        first_err = next((r for r in records if isinstance(r, dict) and r.get("error")), {})
        return {
            "symbol": symbol,
            "error": str(first_err.get("error") or "InvocationError"),
            "message": str(first_err.get("message") or "SageMaker returned no usable records"),
        }

    score, label, analyzed = aggregate_predictions(ok_records)

    src_counts: dict[str, int] = {"news_rss": 0, "reddit": 0}
    for it in kept:
        src_counts[it.source_type] = src_counts.get(it.source_type, 0) + 1

    out = {
        "symbol": symbol,
        "score": round(score, 6),
        "label": label,
        "article_count": analyzed,
        "recent_headlines": recent_headlines(kept, RECENT_HEADLINES_MAX),
        "sources": src_counts,
        "updated_at": int(time.time()),
    }
    _enqueue_cache_write(out)
    return out


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """Route direct Lambda invoke vs HTTP POST; map SageMaker errors to HTTP status codes."""
    # Direct invocation: { "symbol": "AAPL", "options": { ... } }
    if "requestContext" not in event:
        raw_sym = event.get("symbol")
        sym = normalize_symbol(str(raw_sym) if raw_sym is not None else "")
        if not sym:
            return {"error": "invalid_symbol", "message": "Provide a valid symbol string"}
        if not _is_known_symbol(sym):
            return {"error": "invalid_symbol", "message": f"Unknown ticker symbol: {sym}"}
        return run_sentiment(sym, event.get("options") if isinstance(event.get("options"), dict) else {})

    if event.get("requestContext", {}).get("http", {}).get("method") != "POST":
        return response(405, {"error": "method_not_allowed"})

    payload = parse_json_body(event)
    raw_sym = payload.get("symbol")
    sym = normalize_symbol(str(raw_sym) if raw_sym is not None else "")
    if not sym:
        return response(400, {"error": "invalid_symbol", "message": "Body must include symbol"})
    if not _is_known_symbol(sym):
        return response(400, {"error": "invalid_symbol", "message": f"Unknown ticker symbol: {sym}"})

    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    out = run_sentiment(sym, options)

    if out.get("error"):
        ec = str(out.get("error", ""))
        if ec == "invalid_symbol":
            return response(400, out)
        if ec in _5XX_ERRORS:
            return response(502, out)
        return response(500, out)
    return response(200, out)
