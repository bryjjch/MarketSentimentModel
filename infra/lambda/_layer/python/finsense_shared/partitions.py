"""Hive-style S3 key helpers so raw/predictions/pseudo/curated share the same layout.

Layout:
  raw/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl
  predictions/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl
  pseudo/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl
  curated/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl

All writers use JSON Lines so downstream consumers (Athena, SageMaker training,
Glue crawlers) can append new partitions without reprocessing history.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

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
