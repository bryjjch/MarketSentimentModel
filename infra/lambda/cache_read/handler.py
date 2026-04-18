"""GET /sentiment/cache/{symbol} � read precomputed snapshot from DynamoDB."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any

import boto3

TABLE_NAME = os.environ["TABLE_NAME"]
_table = boto3.resource("dynamodb").Table(TABLE_NAME)

_JSON = {"Content-Type": "application/json"}


def _json_safe(obj: Any) -> Any:
    """Helper function to convert the object to a JSON safe object."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    return obj


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """Lambda handler for the cache read."""
    sym = (event.get("pathParameters") or {}).get("symbol") or ""
    sym = sym.strip().upper()
    if not sym:
        return {
            "statusCode": 400,
            "headers": _JSON,
            "body": json.dumps({"error": "missing_symbol"}),
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
