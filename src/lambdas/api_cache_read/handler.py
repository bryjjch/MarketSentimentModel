"""GET /sentiment/cache — list active cache rows; GET /sentiment/cache/{symbol} — one row.

Read-only view of the DynamoDB ``sentiment_cache`` table (writes are owned by the
cache_write Lambda; ticker autocomplete lives in api_ticker_suggest).
"""

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

from finsense_shared.http import json_safe, parse_limit, query_params, response

TABLE_NAME = os.environ["TABLE_NAME"]
_table = boto3.resource("dynamodb").Table(TABLE_NAME)

_DEFAULT_LIST_LIMIT = 100
_MAX_LIST_LIMIT = 500


def _encode_cursor(key: dict[str, Any]) -> str:
    """Encode DynamoDB LastEvaluatedKey to a URL-safe cursor string."""
    payload = json.dumps(json_safe(key), separators=(",", ":"), sort_keys=True).encode("utf-8")
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
    sym = (event.get("pathParameters") or {}).get("symbol") or ""
    sym = sym.strip().upper()
    if not sym:
        now = int(time.time())
        query = query_params(event)
        try:
            limit = parse_limit(query, default=_DEFAULT_LIST_LIMIT, maximum=_MAX_LIST_LIMIT)
            cursor = _decode_cursor(query.get("cursor"))
        except ValueError as exc:
            return response(400, {"error": "bad_request", "message": str(exc)})
        rows, next_key = _scan_active_items(now, limit, cursor)
        extra_headers = {"X-Next-Cursor": _encode_cursor(next_key)} if next_key else None
        return response(200, [json_safe(item) for item in rows], extra_headers)

    resp = _table.get_item(Key={"symbol": sym})
    item = resp.get("Item")
    if not item:
        return response(404, {"error": "not_found", "symbol": sym})

    expires_at = item.get("expires_at")
    if expires_at is not None and int(expires_at) <= int(time.time()):
        return response(404, {"error": "not_found", "symbol": sym})

    return response(200, json_safe(item))
