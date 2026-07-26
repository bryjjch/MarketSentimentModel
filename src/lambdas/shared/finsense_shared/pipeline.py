"""The contracts every pipeline stage agrees on: S3 keys, SQS tasks, and the cache item.

Stages never import each other — they only agree on the shapes defined here, so a
change to a payload or key layout has exactly one place to happen.

**S3 layout** — Hive-style partitions shared by raw/predictions/pseudo/curated::

    raw/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl
    predictions/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl
    pseudo/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl
    curated/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl

All writers use JSON Lines so downstream consumers (Athena, SageMaker training, Glue
crawlers) can append new partitions without reprocessing history.

**SQS tasks** — one builder + one validator per queue keeps producers and consumers in
lockstep:

* collect queue     — ``pipeline_dispatch`` -> ``pipeline_collect``
* predict queue     — ``pipeline_collect`` -> ``pipeline_predict``
* label queue       — ``pipeline_predict`` -> ``pipeline_label`` (pointer form: the
  low-confidence rows are re-read from the predictions object by ``row_index`` instead
  of being inlined, so payloads never approach the 256 KB SQS limit)
* cache-write queue — ``pipeline_predict`` and ``api_sentiment`` -> ``cache_write``

**Cache item** — the DynamoDB row ``cache_write`` builds from a cache-write task.
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# S3 partition layout
# ---------------------------------------------------------------------------

RAW_PREFIX = "raw"
PREDICTIONS_PREFIX = "predictions"
PSEUDO_PREFIX = "pseudo"
CURATED_PREFIX = "curated"


def _dt_str(when: date | datetime | None) -> str:
    if when is None:
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    if isinstance(when, datetime):
        return when.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return when.strftime("%Y-%m-%d")


def partition_prefix(
    dataset: str,
    *,
    symbol: str,
    when: date | datetime | None = None,
) -> str:
    """Return ``<dataset>/dt=YYYY-MM-DD/symbol=SYM/`` (no leading slash, trailing slash included)."""
    return f"{dataset}/dt={_dt_str(when)}/symbol={symbol.upper()}/"


def _key(dataset: str, *, symbol: str, run_id: str, when: date | datetime | None) -> str:
    return f"{partition_prefix(dataset, symbol=symbol, when=when)}{run_id}.jsonl"


def raw_key(symbol: str, run_id: str, *, when: date | datetime | None = None) -> str:
    """Key for the raw ingestion payload for a given symbol + run."""
    return _key(RAW_PREFIX, symbol=symbol, run_id=run_id, when=when)


def prediction_key(symbol: str, run_id: str, *, when: date | datetime | None = None) -> str:
    """Key for per-text prediction records (incl. probabilities + confidence)."""
    return _key(PREDICTIONS_PREFIX, symbol=symbol, run_id=run_id, when=when)


def pseudo_label_key(symbol: str, run_id: str, *, when: date | datetime | None = None) -> str:
    """Key for LLM pseudo-labels of low-confidence rows."""
    return _key(PSEUDO_PREFIX, symbol=symbol, run_id=run_id, when=when)


def curated_key(symbol: str, run_id: str, *, when: date | datetime | None = None) -> str:
    """Key for curated training rows (combination of high-confidence + pseudo-labeled)."""
    return _key(CURATED_PREFIX, symbol=symbol, run_id=run_id, when=when)


_DT_RE = re.compile(r"/dt=(\d{4}-\d{2}-\d{2})/")


def dt_from_key(key: str) -> date | None:
    """Extract the ``dt`` partition date from a Hive-style S3 key.

    Returns a :class:`datetime.date` when the key contains ``/dt=YYYY-MM-DD/``,
    or ``None`` if the pattern is absent or unparseable.
    """
    m = _DT_RE.search(key)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# SQS task messages
# ---------------------------------------------------------------------------

TASK_COLLECT = "collect"
TASK_PREDICT = "predict"
TASK_LABEL = "label"
TASK_CACHE_WRITE = "cache_write"


def build_collect_task(
    run_id: str,
    symbol: str,
    *,
    max_articles: int,
    include_social: bool,
) -> dict[str, Any]:
    return {
        "task": TASK_COLLECT,
        "run_id": run_id,
        "symbol": symbol,
        "options": {
            "max_articles": int(max_articles),
            "include_social": bool(include_social),
        },
    }


def build_predict_task(
    run_id: str,
    symbol: str,
    *,
    bucket: str,
    key: str,
    count: int,
) -> dict[str, Any]:
    return {
        "task": TASK_PREDICT,
        "run_id": run_id,
        "symbol": symbol,
        "bucket": bucket,
        "key": key,
        "count": int(count),
    }


def build_label_task(
    run_id: str,
    symbol: str,
    *,
    bucket: str,
    dt: str | None,
    predictions_key: str,
    row_indices: list[int],
) -> dict[str, Any]:
    return {
        "task": TASK_LABEL,
        "run_id": run_id,
        "symbol": symbol,
        "bucket": bucket,
        "dt": dt,
        "predictions_key": predictions_key,
        "row_indices": [int(i) for i in row_indices],
    }


def build_cache_write_task(
    symbol: str,
    *,
    score: float,
    label: str,
    article_count: int,
    recent_headlines: list[dict[str, str]],
    updated_at: int,
    ttl_seconds: int,
    source: str,
) -> dict[str, Any]:
    return {
        "task": TASK_CACHE_WRITE,
        "symbol": symbol,
        "score": score,
        "label": label,
        "article_count": int(article_count),
        "recent_headlines": recent_headlines,
        "updated_at": int(updated_at),
        "ttl_seconds": int(ttl_seconds),
        "source": source,
    }


_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    TASK_COLLECT: ("run_id", "symbol"),
    TASK_PREDICT: ("run_id", "symbol", "bucket", "key"),
    TASK_LABEL: ("run_id", "symbol", "bucket", "predictions_key", "row_indices"),
    TASK_CACHE_WRITE: ("symbol", "score", "label", "updated_at", "ttl_seconds"),
}


def validate_task(payload: Any, expected_task: str) -> dict[str, Any]:
    """Return ``payload`` if it is a well-formed task dict, else raise ``ValueError``."""
    if not isinstance(payload, dict):
        raise ValueError(f"{expected_task} task must be a JSON object")
    if payload.get("task") != expected_task:
        raise ValueError(f"expected task={expected_task!r}, got {payload.get('task')!r}")
    missing = [f for f in _REQUIRED_FIELDS[expected_task] if payload.get(f) in (None, "")]
    if missing:
        raise ValueError(f"{expected_task} task missing fields: {', '.join(missing)}")
    return payload


# ---------------------------------------------------------------------------
# DynamoDB cache item
# ---------------------------------------------------------------------------


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
