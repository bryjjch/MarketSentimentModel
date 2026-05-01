"""HTTP API (v2) handler: POST /sentiment/by-symbol (symbol to sources to SageMaker to aggregate)."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import boto3
from botocore.exceptions import ClientError

from finsense_shared import aggregate_predictions, load_valid_ticker_set, normalize_symbol
from finsense_shared.sources import collect_for_symbol
from finsense_shared.sources.base import CollectedItem

ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
RECENT_HEADLINES_MAX = int(os.environ.get("RECENT_HEADLINES_MAX", "10"))
DEFAULT_MAX_ARTICLES = int(os.environ.get("DEFAULT_MAX_ARTICLES", "12"))
CACHE_TABLE_NAME = os.environ.get("CACHE_TABLE_NAME", "").strip()
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "604800"))

_runtime = boto3.client("sagemaker-runtime")
_ddb = boto3.resource("dynamodb") if CACHE_TABLE_NAME else None
_JSON_HEADERS = {"Content-Type": "application/json"}
logger = logging.getLogger()


def _is_known_symbol(symbol: str) -> bool:
    return symbol in load_valid_ticker_set()


def _to_ddb_number(value: Any, default: str = "0") -> Decimal:
    """Convert numeric-like values to Decimal for DynamoDB writes."""
    try:
        d = Decimal(str(value))
        if d.is_nan() or d.is_infinite():
            return Decimal(default)
        return d
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _write_cache_row(
    symbol: str,
    score: float,
    label: str,
    article_count: int,
    recent_headlines: list[dict[str, str]],
) -> None:
    """Best-effort cache write so long-tail symbols are reused by cache reads."""
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
                "article_count": int(article_count),
                "recent_headlines": recent_headlines,
                "updated_at": now,
                "expires_at": now + CACHE_TTL_SECONDS,
            }
        )
    except ClientError as e:
        logger.exception("ddb_put_failed %s: %s", symbol, e)


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    """HTTP API v2 proxy response shape (JSON body)."""
    return {
        "statusCode": status,
        "headers": _JSON_HEADERS,
        "body": json.dumps(body),
    }


def _parse_api_body(event: dict[str, Any]) -> dict[str, Any]:
    """Decode optional base64 body and parse JSON; invalid JSON yields ``{}``."""
    raw_body = event.get("body")
    if raw_body in (None, ""):
        return {}
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw_body).decode("utf-8")
    else:
        raw = raw_body if isinstance(raw_body, str) else raw_body.decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def run_sentiment(symbol: str, options: dict[str, Any] | None) -> dict[str, Any]:
    """Core pipeline: collect texts, invoke SageMaker, aggregate. Used by API and direct invoke."""
    opts = options or {}
    max_articles = int(opts.get("max_articles", DEFAULT_MAX_ARTICLES))
    max_articles = max(1, min(max_articles, 40))
    include_social = bool(opts.get("include_social", True))

    items: list[CollectedItem] = collect_for_symbol(
        symbol,
        max_articles=max_articles,
        include_social=include_social,
    )

    if not items:
        now = int(time.time())
        return {
            "symbol": symbol,
            "score": 0.0,
            "label": "neutral",
            "article_count": 0,
            "recent_headlines": [],
            "sources": {"news_rss": 0, "reddit": 0},
            "updated_at": now,
            "detail": "no_articles_collected",
        }

    # Drop empty snippets so SageMaker batch aligns with kept items
    kept: list[CollectedItem] = [it for it in items if it.text.strip()]
    if not kept:
        now = int(time.time())
        return {
            "symbol": symbol,
            "score": 0.0,
            "label": "neutral",
            "article_count": 0,
            "recent_headlines": [],
            "sources": {"news_rss": 0, "reddit": 0},
            "updated_at": now,
            "detail": "no_non_empty_text",
        }

    texts = [it.text for it in kept]
    payload = json.dumps({"texts": texts}).encode("utf-8")

    try:
        sm_resp = _runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
    except ClientError as e:
        err = e.response.get("Error", {})
        return {
            "symbol": symbol,
            "error": err.get("Code", "InvocationError"),
            "message": err.get("Message", str(e)),
        }

    raw_out = sm_resp["Body"].read().decode("utf-8")
    try:
        records: list[dict[str, Any]] = json.loads(raw_out)
    except json.JSONDecodeError:
        return {"symbol": symbol, "error": "invalid_sagemaker_response"}

    if not isinstance(records, list):
        return {"symbol": symbol, "error": "invalid_sagemaker_shape"}

    if len(records) != len(texts):
        return {"symbol": symbol, "error": "sagemaker_batch_size_mismatch"}

    score, label, analyzed = aggregate_predictions(records)

    headlines: list[dict[str, str]] = []
    for it in kept[:RECENT_HEADLINES_MAX]:
        headlines.append({"title": it.title, "url": it.url})

    src_counts: dict[str, int] = {"news_rss": 0, "reddit": 0}
    for it in kept:
        if it.source_type in src_counts:
            src_counts[it.source_type] += 1
        else:
            src_counts[it.source_type] = src_counts.get(it.source_type, 0) + 1

    now = int(time.time())
    out = {
        "symbol": symbol,
        "score": round(score, 6),
        "label": label,
        "article_count": analyzed,
        "recent_headlines": headlines,
        "sources": src_counts,
        "updated_at": now,
    }
    _write_cache_row(
        symbol=symbol,
        score=out["score"],
        label=label,
        article_count=analyzed,
        recent_headlines=headlines,
    )
    return out


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """Route direct Lambda invoke vs HTTP POST; map SageMaker errors to HTTP status codes."""
    # Direct invocation (refresher): { "symbol": "AAPL", "options": { ... } }
    if "requestContext" not in event:
        raw_sym = event.get("symbol")
        sym = normalize_symbol(str(raw_sym) if raw_sym is not None else "")
        if not sym:
            return {"error": "invalid_symbol", "message": "Provide a valid symbol string"}
        if not _is_known_symbol(sym):
            return {"error": "invalid_symbol", "message": f"Unknown ticker symbol: {sym}"}
        out = run_sentiment(sym, event.get("options") if isinstance(event.get("options"), dict) else {})
        return out

    if event.get("requestContext", {}).get("http", {}).get("method") != "POST":
        return _response(405, {"error": "method_not_allowed"})

    payload = _parse_api_body(event)
    raw_sym = payload.get("symbol")
    sym = normalize_symbol(str(raw_sym) if raw_sym is not None else "")
    if not sym:
        return _response(400, {"error": "invalid_symbol", "message": "Body must include symbol"})
    if not _is_known_symbol(sym):
        return _response(400, {"error": "invalid_symbol", "message": f"Unknown ticker symbol: {sym}"})

    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    out = run_sentiment(sym, options)

    if out.get("error"):
        ec = str(out.get("error", ""))
        if ec == "invalid_symbol":
            return _response(400, out)
        if ec in ("ModelError", "InternalFailure", "InvocationError", "ServiceUnavailable"):
            return _response(502, out)
        return _response(500, out)
    return _response(200, out)
