"""GET /sentiment/cache — list active cache rows; GET /sentiment/cache/{symbol} — one row."""

from __future__ import annotations

import base64
import binascii
import json
import os
import time
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr
from finsense_shared import search_tickers_by_prefix

TABLE_NAME = os.environ["TABLE_NAME"]
_table = boto3.resource("dynamodb").Table(TABLE_NAME)

_JSON = {"Content-Type": "application/json"}
_DEFAULT_LIST_LIMIT = 100
_MAX_LIST_LIMIT = 500
_DEFAULT_SUGGEST_LIMIT = 10
_MAX_SUGGEST_LIMIT = 25


def _json_safe(obj: Any) -> Any:
    """Recursively convert DynamoDB types (e.g. ``Decimal``) for ``json.dumps``."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    return obj


def _parse_list_limit(query: dict[str, str] | None) -> int:
    """Parse and clamp requested list size."""
    raw = (query or {}).get("limit")
    if raw is None or raw == "":
        return _DEFAULT_LIST_LIMIT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer") from None
    if value < 1:
        raise ValueError("limit must be >= 1")
    return min(value, _MAX_LIST_LIMIT)


def _encode_cursor(key: dict[str, Any]) -> str:
    """Encode DynamoDB LastEvaluatedKey to a URL-safe cursor string."""
    payload = json.dumps(_json_safe(key), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(raw: str | None) -> dict[str, Any] | None:
    """Decode URL cursor back into DynamoDB ExclusiveStartKey."""
    if not raw:
        return None
    try:
        padding = "=" * (-len(raw) % 4)
        data = base64.urlsafe_b64decode((raw + padding).encode("ascii"))
        obj = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error):
        raise ValueError("cursor is invalid") from None
    if not isinstance(obj, dict):
        raise ValueError("cursor is invalid")
    return obj


def _parse_suggest_limit(query: dict[str, str] | None) -> int:
    """Parse and clamp requested suggest size."""
    raw = (query or {}).get("limit")
    if raw is None or raw == "":
        return _DEFAULT_SUGGEST_LIMIT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer") from None
    if value < 1:
        raise ValueError("limit must be >= 1")
    return min(value, _MAX_SUGGEST_LIMIT)


def _is_suggest_route(event: dict[str, Any]) -> bool:
    """Check if the request is for the ticker suggestions endpoint."""
    path = ((event.get("requestContext") or {}).get("http") or {}).get("path") or ""
    raw_path = event.get("rawPath") or ""
    return path.endswith("/tickers/suggest") or raw_path.endswith("/tickers/suggest")


def _scan_active_items(now: int, limit: int, cursor: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Scan one page of active rows and return a pagination key when available."""
    kwargs: dict[str, Any] = {
        "FilterExpression": Attr("expires_at").not_exists() | Attr("expires_at").gt(Decimal(now)),
        "Limit": limit,
    }
    if cursor:
        kwargs["ExclusiveStartKey"] = cursor
    resp = _table.scan(**kwargs)
    items = resp.get("Items", [])
    return items, resp.get("LastEvaluatedKey")


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """One row for ``pathParameters.symbol``, or all active rows when symbol is omitted."""
    if _is_suggest_route(event):
        query = event.get("queryStringParameters") or {}
        q = (query.get("q") or "").strip()
        if not q:
            return {
                "statusCode": 400,
                "headers": _JSON,
                "body": json.dumps({"error": "bad_request", "message": "q is required"}),
            }
        try:
            limit = _parse_suggest_limit(query)
        except ValueError as exc:
            return {
                "statusCode": 400,
                "headers": _JSON,
                "body": json.dumps({"error": "bad_request", "message": str(exc)}),
            }
        suggestions = search_tickers_by_prefix(q, limit=limit)
        return {
            "statusCode": 200,
            "headers": _JSON,
            "body": json.dumps({"query": q.upper(), "suggestions": suggestions}),
        }

    sym = (event.get("pathParameters") or {}).get("symbol") or ""
    sym = sym.strip().upper()
    if not sym:
        now = int(time.time())
        query = event.get("queryStringParameters") or {}
        try:
            limit = _parse_list_limit(query)
            cursor = _decode_cursor(query.get("cursor"))
        except ValueError as exc:
            return {
                "statusCode": 400,
                "headers": _JSON,
                "body": json.dumps({"error": "bad_request", "message": str(exc)}),
            }
        rows, next_key = _scan_active_items(now, limit, cursor)
        headers = _JSON.copy()
        if next_key:
            headers["X-Next-Cursor"] = _encode_cursor(next_key)
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps([_json_safe(item) for item in rows]),
        }

    resp = _table.get_item(Key={"symbol": sym})
    item = resp.get("Item")
    if not item:
        return {
            "statusCode": 404,
            "headers": _JSON,
            "body": json.dumps({"error": "not_found", "symbol": sym}),
        }

    expires_at = item.get("expires_at")
    if expires_at is not None and int(expires_at) <= int(time.time()):
        return {
            "statusCode": 404,
            "headers": _JSON,
            "body": json.dumps({"error": "not_found", "symbol": sym}),
        }

    return {
        "statusCode": 200,
        "headers": _JSON,
        "body": json.dumps(_json_safe(item)),
    }
