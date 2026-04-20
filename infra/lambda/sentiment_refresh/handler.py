"""EventBridge schedule: refresh DynamoDB cache by invoking the sentiment orchestration Lambda per ticker."""

from __future__ import annotations

import json
import logging
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SENTIMENT_FUNCTION_NAME = os.environ["SENTIMENT_FUNCTION_NAME"]
TABLE_NAME = os.environ["TABLE_NAME"]
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "604800"))
TOP_TICKERS_SSM_PARAM = os.environ.get("TOP_TICKERS_SSM_PARAM", "").strip()
DEFAULT_TICKERS_JSON = os.environ.get("DEFAULT_TICKERS_JSON", '["AAPL","MSFT","GOOGL"]')

_lambda = boto3.client("lambda")
_ssm = boto3.client("ssm")
_ddb = boto3.resource("dynamodb")


def _to_ddb_number(value: Any, default: str = "0") -> Decimal:
    """Convert numerics to DynamoDB-safe Decimal values."""
    try:
        d = Decimal(str(value))
        if d.is_nan() or d.is_infinite():
            return Decimal(default)
        return d
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _load_tickers() -> list[str]:
    """Resolve ticker list from SSM JSON (if configured) with JSON/env fallbacks."""
    if TOP_TICKERS_SSM_PARAM:
        try:
            r = _ssm.get_parameter(Name=TOP_TICKERS_SSM_PARAM)
            raw = r["Parameter"]["Value"]
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x).strip().upper() for x in data if str(x).strip()]
        except (ClientError, json.JSONDecodeError, KeyError) as e:
            logger.warning("ssm_tickers_failed: %s", e)
    try:
        data = json.loads(DEFAULT_TICKERS_JSON)
        if isinstance(data, list):
            return [str(x).strip().upper() for x in data if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return ["AAPL", "MSFT", "GOOGL"]


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """Invoke sentiment Lambda per ticker and refresh DynamoDB cache rows."""
    tickers = _load_tickers()
    table = _ddb.Table(TABLE_NAME)
    now = int(time.time())
    expires_at = now + CACHE_TTL_SECONDS
    ok = 0
    failed = 0

    for sym in tickers:
        if not sym:
            continue
        payload = json.dumps({"symbol": sym, "options": {}}).encode("utf-8")
        try:
            out = _lambda.invoke(
                FunctionName=SENTIMENT_FUNCTION_NAME,
                InvocationType="RequestResponse",
                Payload=payload,
            )
            body_raw = out["Payload"].read().decode("utf-8")
            body = json.loads(body_raw) if body_raw else {}
        except (ClientError, json.JSONDecodeError) as e:
            logger.exception("invoke_failed %s: %s", sym, e)
            failed += 1
            continue

        if body.get("error"):
            logger.warning("sentiment_error %s: %s", sym, body.get("error"))
            failed += 1
            continue

        try:
            table.put_item(
                Item={
                    "symbol": sym,
                    "score": _to_ddb_number(body.get("score", 0)),
                    "label": body.get("label", "neutral"),
                    "article_count": int(body.get("article_count", 0)),
                    "recent_headlines": body.get("recent_headlines", []),
                    "updated_at": int(body.get("updated_at", now)),
                    "expires_at": int(expires_at),
                }
            )
            ok += 1
        except ClientError as e:
            logger.exception("ddb_put_failed %s: %s", sym, e)
            failed += 1

    result = {"tickers": len(tickers), "ok": ok, "failed": failed, "ts": now}
    logger.info("refresh_complete %s", result)
    return result
