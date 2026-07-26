"""Shared primitives copied into every FinSense Lambda container image.

Layout:

* ``pipeline``   — S3 key layout, SQS task shapes, DynamoDB cache item: the contracts
  the stages agree on.
* ``sentiment``  — reading class probabilities: confidence gating and aggregation.
* ``tickers/``   — symbol syntax, the ticker universe, company names, bundled JSON.
* ``sources/``   — news and social adapters that produce text to score.
* ``aws/``       — thin boto3 wrappers (S3, SQS, SageMaker runtime).
* ``http``       — API Gateway request/response helpers.
* ``llm_label``  — provider-agnostic pseudo-labeler for low-confidence rows.

The names below are the common ones, re-exported flat for handler convenience; anything
else is imported from its module (e.g. ``from finsense_shared.aws.s3 import write_jsonl``).
"""

from __future__ import annotations

from .pipeline import (
    build_cache_item,
    curated_key,
    dt_from_key,
    partition_prefix,
    prediction_key,
    pseudo_label_key,
    raw_key,
    recent_headlines,
    to_ddb_number,
)
from .sentiment import (
    ConfidenceMetric,
    aggregate_predictions,
    confidence_from_probabilities,
    is_low_confidence,
    sentiment_score_from_probabilities,
)
from .tickers import (
    load_tickers,
    load_valid_ticker_set,
    load_valid_tickers,
    normalize_symbol,
    search_tickers_by_prefix,
)

__all__ = [
    "ConfidenceMetric",
    "aggregate_predictions",
    "build_cache_item",
    "confidence_from_probabilities",
    "curated_key",
    "dt_from_key",
    "is_low_confidence",
    "load_tickers",
    "load_valid_ticker_set",
    "load_valid_tickers",
    "normalize_symbol",
    "partition_prefix",
    "prediction_key",
    "pseudo_label_key",
    "raw_key",
    "recent_headlines",
    "search_tickers_by_prefix",
    "sentiment_score_from_probabilities",
    "to_ddb_number",
]
