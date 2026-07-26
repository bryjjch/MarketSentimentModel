"""HTTP API (v2) request/response helpers shared by the API Lambdas."""

from __future__ import annotations

import base64
import json
from decimal import Decimal
from typing import Any

_JSON_HEADERS = {"Content-Type": "application/json"}


def json_safe(obj: Any) -> Any:
    """Recursively convert DynamoDB types (e.g. ``Decimal``) for ``json.dumps``."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    return obj


def response(status: int, body: Any, extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
    """HTTP API v2 proxy response shape (JSON body)."""
    headers = dict(_JSON_HEADERS)
    if extra_headers:
        headers.update(extra_headers)
    return {
        "statusCode": status,
        "headers": headers,
        "body": json.dumps(json_safe(body)),
    }


def parse_json_body(event: dict[str, Any]) -> dict[str, Any]:
    """Decode optional base64 body and parse JSON; invalid JSON yields ``{}``."""
    raw_body = event.get("body")
    if raw_body in (None, ""):
        return {}
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw_body).decode("utf-8")
    else:
        raw = raw_body if isinstance(raw_body, str) else raw_body.decode("utf-8")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def query_params(event: dict[str, Any]) -> dict[str, str]:
    return event.get("queryStringParameters") or {}


def parse_limit(query: dict[str, str] | None, *, default: int, maximum: int) -> int:
    """Parse and clamp a ``limit`` query parameter. Raises ``ValueError`` on bad input."""
    raw = (query or {}).get("limit")
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer") from None
    if value < 1:
        raise ValueError("limit must be >= 1")
    return min(value, maximum)
