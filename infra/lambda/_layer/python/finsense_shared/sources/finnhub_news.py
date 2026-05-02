"""Finnhub company news: symbol-keyed articles (avoids generic web-search false positives)."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Any

from .base import CollectedItem

logger = logging.getLogger(__name__)

_FINNHUB_COMPANY_NEWS = "https://finnhub.io/api/v1/company-news"
_TAG_RE = re.compile(r"<[^>]+>")

# Cache one resolved key per Lambda execution environment (single Secrets Manager read per
# warm container; ``_collect_news_slot`` passes the key into ``collect_finnhub_news`` to
# skip a second lookup).
_cached_fp: str | None = None


def _finnhub_env_fingerprint() -> str:
    return (
        f"{os.environ.get('FINNHUB_API_KEY', '')}\0"
        f"{os.environ.get('FINNHUB_SECRET_ARN', '')}"
    )


def _resolve_finnhub_api_key_uncached() -> str | None:
    """Read API key from env or Secrets Manager (no caching)."""
    direct = os.environ.get("FINNHUB_API_KEY", "").strip()
    if direct:
        return direct
    arn = os.environ.get("FINNHUB_SECRET_ARN", "").strip()
    if not arn:
        return None
    try:
        import boto3

        client = boto3.client("secretsmanager")
        resp = client.get_secret_value(SecretId=arn)
        raw = resp.get("SecretString") or ""
        data: Any
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return str(raw).strip() or None
        if isinstance(data, dict):
            k = str(data.get("api_key", "") or data.get("token", "") or "").strip()
            return k or None
        if isinstance(data, str):
            return data.strip() or None
    except Exception as e:  # noqa: BLE001
        logger.warning("finnhub_secret_read_failed: %s", e)
    return None


_MISSING = object()
_cached_key: str | None | object = _MISSING


def get_finnhub_api_key() -> str | None:
    """Return Finnhub API key once per env configuration (cached across symbols in one invocation)."""
    global _cached_fp, _cached_key
    fp = _finnhub_env_fingerprint()
    if _cached_key is not _MISSING and fp == _cached_fp:
        return _cached_key  # type: ignore[return-value]
    _cached_fp = fp
    key = _resolve_finnhub_api_key_uncached()
    _cached_key = key
    return key


def finnhub_news_enabled() -> bool:
    """True when Finnhub API key is available (env or Secrets Manager)."""
    return bool(get_finnhub_api_key())


def clear_finnhub_api_key_cache_for_tests() -> None:
    """Reset module cache (tests only)."""
    global _cached_fp, _cached_key
    _cached_fp = None
    _cached_key = _MISSING


def _strip_html(s: str) -> str:
    """Strip HTML tags and unescape entities."""
    t = _TAG_RE.sub(" ", s)
    t = unescape(t)
    return " ".join(t.split()).strip()


def _related_has_symbol(related: str, symbol: str) -> bool:
    """Check if the related symbols contain the given symbol."""
    if not related or not symbol:
        return False
    sym = symbol.upper()
    for part in related.split(","):
        if part.strip().upper() == sym:
            return True
    return False


def _fetch_company_news(
    symbol: str,
    token: str,
    *,
    from_date: str,
    to_date: str,
    timeout_s: float = 12.0,
) -> list[dict[str, Any]]:
    """Fetch company news from Finnhub."""
    q = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": token,
        }
    )
    url = f"{_FINNHUB_COMPANY_NEWS}?{q}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FinSense/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.HTTPError) as e:
        logger.warning("finnhub_http_error %s: %s", symbol, e)
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("finnhub_invalid_json %s", symbol)
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def collect_finnhub_news(
    symbol: str,
    *,
    max_items: int,
    api_key: str | None = None,
) -> list[CollectedItem]:
    """Return company news articles from Finnhub keyed to ``symbol`` (up to ``max_items``).

    Pass ``api_key`` from :func:`get_finnhub_api_key` when resolving once per batch to avoid
    a second cache lookup. When ``api_key`` is omitted, resolves via :func:`get_finnhub_api_key`.

    Empty list if no API key, HTTP/JSON failure, or no articles in range.
    """
    token = api_key if api_key is not None else get_finnhub_api_key()
    if not token:
        return []

    max_items = max(1, max_items)
    # Finnhub returns latest within range; use last 30 UTC calendar days for coverage.
    to_d = datetime.now(timezone.utc).date()
    from_d = to_d - timedelta(days=30)

    raw_list = _fetch_company_news(
        symbol,
        token,
        from_date=from_d.isoformat(),
        to_date=to_d.isoformat(),
    )
    if not raw_list:
        return []

    sym_u = symbol.upper()
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in raw_list:
        related = str(row.get("related") or "")
        if not _related_has_symbol(related, sym_u):
            continue
        dt_raw = row.get("datetime")
        ts = 0
        if isinstance(dt_raw, (int, float)):
            ts = int(dt_raw)
        elif isinstance(dt_raw, str) and dt_raw.isdigit():
            ts = int(dt_raw)
        if ts > 1_000_000_000_000:
            ts //= 1000
        scored.append((ts, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    items: list[CollectedItem] = []
    for _ts, row in scored:
        if len(items) >= max_items:
            break
        headline = str(row.get("headline") or "").strip()
        summary = _strip_html(str(row.get("summary") or ""))
        url = str(row.get("url") or "").strip()
        source = str(row.get("source") or "").strip()
        title = headline or url or symbol
        text_parts = [headline, summary, f"Source: {source}" if source else ""]
        text = "\n".join(p for p in text_parts if p).strip()[:2000]
        if not text:
            continue
        items.append(
            CollectedItem(
                title=title,
                url=url or f"https://finnhub.io/stock/{urllib.parse.quote(sym_u)}",
                text=text,
                source_type="finnhub",
            )
        )
    return items
