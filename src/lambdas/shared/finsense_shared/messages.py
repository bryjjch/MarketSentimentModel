"""Task message schemas passed between pipeline stages over SQS.

One builder + one validator per queue keeps producers and consumers in lockstep:

* collect queue   — ``pipeline_dispatch`` -> ``pipeline_collect``
* predict queue   — ``pipeline_collect`` -> ``pipeline_predict``
* label queue     — ``pipeline_predict`` -> ``pipeline_label`` (pointer form: the
  low-confidence rows are re-read from the predictions object by ``row_index``
  instead of being inlined, so payloads never approach the 256 KB SQS limit)
* cache-write queue — ``pipeline_predict`` and ``api_sentiment`` -> ``cache_write``
"""

from __future__ import annotations

from typing import Any

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
