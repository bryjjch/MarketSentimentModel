"""Reading the model's class probabilities: how confident, and what the symbol scores.

Both halves consume the same ``{"negative": .., "neutral": .., "positive": ..}`` dict
the SageMaker endpoint returns per text:

* :func:`confidence_from_probabilities` / :func:`is_low_confidence` judge a single
  prediction. A row is low-confidence when the top-class probability is below a
  threshold (default 0.65) *or*, optionally, when the margin between top and runner-up
  is below a margin threshold. Either trigger routes the row to the pseudo-label Lambda.
* :func:`aggregate_predictions` rolls a symbol's rows up into the score and label the
  API serves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Per-prediction confidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceMetric:
    """Structured confidence snapshot for a single prediction."""

    top_prob: float
    margin: float
    entropy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "top_prob": round(self.top_prob, 6),
            "margin": round(self.margin, 6),
            "entropy": round(self.entropy, 6),
        }


def confidence_from_probabilities(probs: dict[str, float]) -> ConfidenceMetric:
    """Derive ``top_prob``, ``margin`` (top - runner-up), and Shannon entropy from a class-prob dict."""
    values = [float(v) for v in probs.values() if v is not None]
    if not values:
        return ConfidenceMetric(top_prob=0.0, margin=0.0, entropy=0.0)
    sorted_vals = sorted(values, reverse=True)
    top = sorted_vals[0]
    runner_up = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    margin = max(0.0, top - runner_up)
    entropy = 0.0
    for v in values:
        if v > 0.0:
            entropy -= v * math.log(v)
    return ConfidenceMetric(top_prob=top, margin=margin, entropy=entropy)


def is_low_confidence(
    probs: dict[str, float],
    *,
    min_top_prob: float = 0.65,
    min_margin: float = 0.0,
) -> bool:
    """Return True when the model's top class is uncertain enough to benefit from LLM relabeling.

    ``min_top_prob`` of 0.65 roughly flags rows where the model's winning class has
    less than 65% probability mass; ``min_margin`` 0 disables the margin gate.
    """
    m = confidence_from_probabilities(probs)
    if m.top_prob < min_top_prob:
        return True
    if min_margin > 0.0 and m.margin < min_margin:
        return True
    return False


# ---------------------------------------------------------------------------
# Symbol-level aggregation
# ---------------------------------------------------------------------------


def sentiment_score_from_probabilities(probs: dict[str, float]) -> float:
    """Map class probabilities to a scalar in roughly [-1, 1] (positive minus negative)."""
    return float(probs.get("positive", 0.0)) - float(probs.get("negative", 0.0))


def aggregate_predictions(records: list[dict[str, Any]]) -> tuple[float, str, int]:
    """Return (score, label, analyzed_count) over a symbol's per-text prediction rows."""
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

    if mean_score > 0.12:
        label = "positive"
    elif mean_score < -0.12:
        label = "negative"
    else:
        label = "neutral"

    return mean_score, label, len(valid)
