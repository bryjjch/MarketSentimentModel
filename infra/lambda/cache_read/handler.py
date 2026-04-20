"""GET /sentiment/cache — list active cache rows; GET /sentiment/cache/{symbol} — one row."""

from __future__ import annotations

import json
import os
import time
from decimal import Decimal
from typing import Any

import boto3

TABLE_NAME = os.environ["TABLE_NAME"]
_table = boto3.resource("dynamodb").Table(TABLE_NAME)

_JSON = {"Content-Type": "application/json"}


def _json_safe(obj: Any) -> Any:
    """Recursively convert DynamoDB types (e.g. ``Decimal``) for ``json.dumps``."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    return obj


def _expires_at_int(item: dict[str, Any]) -> int | None:
    """Convert DynamoDB expires_at to integer timestamp."""
    raw = item.get("expires_at")
    if raw is None:
        return None
    if isinstance(raw, Decimal):
        return int(raw)
    return int(raw)


def _is_active(item: dict[str, Any], now: int) -> bool:
    """Check if the item is active (expires_at is in the future)."""
    exp = _expires_at_int(item)
    if exp is None:
        return True
    return exp > now


def _scan_active_items(now: int) -> list[dict[str, Any]]:
    """Scan the table for active items."""
    items: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {}
    while True:
        resp = _table.scan(**kwargs)
        for it in resp.get("Items", []):
            if _is_active(it, now):
                items.append(it)
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break
        kwargs["ExclusiveStartKey"] = lek
    items.sort(key=lambda x: str(x.get("symbol", "")))
    return items


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """One row for ``pathParameters.symbol``, or all active rows when symbol is omitted."""
    sym = (event.get("pathParameters") or {}).get("symbol") or ""
    sym = sym.strip().upper()
    if not sym:
        now = int(time.time())
        rows = _scan_active_items(now)
        return {
            "statusCode": 200,
            "headers": _JSON,
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

    return {
        "statusCode": 200,
        "headers": _JSON,
        "body": json.dumps(_json_safe(item)),
    }
