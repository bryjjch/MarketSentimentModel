"""Aggregate SageMaker per-text predictions into portfolio-level score and label.

Label IDs match training / SageMaker: 0=negative, 1=neutral, 2=positive.
"""

from __future__ import annotations

from typing import Any


def sentiment_score_from_probabilities(probs: dict[str, float]) -> float:
    """Map class probabilities to a scalar in roughly [-1, 1] (positive minus negative)."""
    return float(probs.get("positive", 0.0)) - float(probs.get("negative", 0.0))


def aggregate_predictions(records: list[dict[str, Any]]) -> tuple[float, str, int]:
    """
    Return (score, label, analyzed_count).

    score: mean of (P_positive - P_negative) over successfully analyzed texts.
    label: categorical bucket from that mean (thresholds on score).
    analyzed_count: number of non-error predictions included.
    """
    valid: list[dict[str, Any]] = []
    for r in records:
        if r.get("error"):
            continue
        if not r.get("probabilities"):
            continue
        valid.append(r)

    if not valid:
        return 0.0, "neutral", 0

    scores = [sentiment_score_from_probabilities(r["probabilities"]) for r in valid]
    mean_score = sum(scores) / len(scores)

    # Thresholds on the probability-derived score (same scale as [-1, 1]-ish)
    if mean_score > 0.12:
        label = "positive"
    elif mean_score < -0.12:
        label = "negative"
    else:
        label = "neutral"

    return mean_score, label, len(valid)
