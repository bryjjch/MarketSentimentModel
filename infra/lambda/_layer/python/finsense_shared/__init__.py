"""Shared primitives packaged as a Lambda Layer for Finsense ingestion/prediction/pseudo-label Lambdas.

This package deliberately avoids heavy dependencies (no torch, transformers, openai SDK,
google-generativeai). Everything network-bound uses ``urllib`` + ``boto3`` (provided by the
Lambda runtime) so the layer zip stays small and cold starts fast.
"""

from __future__ import annotations

from .aggregate import aggregate_predictions, sentiment_score_from_probabilities
from .confidence import (
    ConfidenceMetric,
    confidence_from_probabilities,
    is_low_confidence,
)
from .partitions import (
    curated_key,
    partition_prefix,
    prediction_key,
    pseudo_label_key,
    raw_key,
)
from .symbol import normalize_symbol
from .tickers import load_tickers

__all__ = [
    "ConfidenceMetric",
    "aggregate_predictions",
    "confidence_from_probabilities",
    "curated_key",
    "is_low_confidence",
    "load_tickers",
    "normalize_symbol",
    "partition_prefix",
    "prediction_key",
    "pseudo_label_key",
    "raw_key",
    "sentiment_score_from_probabilities",
]
