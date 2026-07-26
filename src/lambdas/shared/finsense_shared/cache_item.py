"""Build the DynamoDB ``sentiment_cache`` item from a cache-write task message."""

from __future__ import annotations

import time
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


def to_ddb_number(value: Any, default: str = "0") -> Decimal:
    """Convert numeric-like values to Decimal for DynamoDB writes."""
    try:
        d = Decimal(str(value))
        if d.is_nan() or d.is_infinite():
            return Decimal(default)
        return d
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def recent_headlines(rows: Iterable[Any], max_n: int) -> list[dict[str, str]]:
    """First ``max_n`` ``{"title", "url"}`` pairs from dicts or attribute-bearing items."""
    out: list[dict[str, str]] = []
    for r in rows:
        if len(out) >= max_n:
            break
        if isinstance(r, dict):
            title, url = r.get("title"), r.get("url")
        else:
            title, url = getattr(r, "title", None), getattr(r, "url", None)
        out.append({"title": title or "", "url": url or ""})
    return out


def build_cache_item(msg: dict[str, Any]) -> dict[str, Any]:
    """Item for ``sentiment_cache`` (hash key ``symbol``, TTL attr ``expires_at``).

    ``ttl_seconds`` travels in the message because the two producers use different
    TTLs (pipeline refresh vs on-demand API write-back).
    """
    updated_at = int(msg.get("updated_at") or time.time())
    ttl_seconds = int(msg["ttl_seconds"])
    return {
        "symbol": str(msg["symbol"]),
        "score": to_ddb_number(msg.get("score")),
        "label": str(msg.get("label") or "neutral"),
        "article_count": int(msg.get("article_count") or 0),
        "recent_headlines": list(msg.get("recent_headlines") or []),
        "updated_at": updated_at,
        "expires_at": updated_at + ttl_seconds,
    }
