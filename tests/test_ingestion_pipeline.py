"""Tests for the shared ingestion/prediction/pseudo-label helpers (``finsense_shared``)."""

from __future__ import annotations

import io
import json

import pytest

from finsense_shared import (
    confidence_from_probabilities,
    curated_key,
    is_low_confidence,
    normalize_symbol,
    partition_prefix,
    prediction_key,
    pseudo_label_key,
    raw_key,
)
from finsense_shared.confidence import ConfidenceMetric
from finsense_shared.llm_label import LABEL_ID_TO_STR, normalize_label, pseudo_label_text


class _FakeS3:
    """Minimal boto3-style S3 client for write_jsonl/read_jsonl round-trips."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, object]] = {}

    def put_object(self, **kwargs: object) -> dict[str, object]:
        key = (str(kwargs["Bucket"]), str(kwargs["Key"]))
        self.objects[key] = dict(kwargs)
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        body = self.objects[(Bucket, Key)]["Body"]
        return {"Body": io.BytesIO(body)}


def test_partition_prefix_hive_style() -> None:
    from datetime import date

    p = partition_prefix("raw", symbol="aapl", when=date(2026, 4, 23))
    assert p == "raw/dt=2026-04-23/symbol=AAPL/"
    assert raw_key("AAPL", "run-1", when=date(2026, 4, 23)) == "raw/dt=2026-04-23/symbol=AAPL/run-1.jsonl"
    assert prediction_key("AAPL", "run-1", when=date(2026, 4, 23)).startswith("predictions/dt=2026-04-23/")
    assert pseudo_label_key("AAPL", "run-1", when=date(2026, 4, 23)).startswith("pseudo/dt=2026-04-23/")
    assert curated_key("AAPL", "run-1", when=date(2026, 4, 23)).startswith("curated/dt=2026-04-23/")


def test_confidence_from_probabilities_balances() -> None:
    uniform = {"negative": 1 / 3, "neutral": 1 / 3, "positive": 1 / 3}
    m = confidence_from_probabilities(uniform)
    assert m.top_prob == pytest.approx(1 / 3)
    assert m.margin == pytest.approx(0.0)
    assert m.entropy == pytest.approx(1.0986, abs=1e-3)

    peaked = {"negative": 0.05, "neutral": 0.05, "positive": 0.9}
    m2 = confidence_from_probabilities(peaked)
    assert m2.top_prob == pytest.approx(0.9)
    assert m2.margin == pytest.approx(0.85)

    assert isinstance(m2, ConfidenceMetric)
    assert "top_prob" in m2.as_dict()


def test_is_low_confidence_thresholds() -> None:
    probs = {"negative": 0.4, "neutral": 0.35, "positive": 0.25}
    assert is_low_confidence(probs, min_top_prob=0.65)
    assert not is_low_confidence({"negative": 0.05, "neutral": 0.05, "positive": 0.9}, min_top_prob=0.65)
    # Margin gate
    assert is_low_confidence({"negative": 0.5, "neutral": 0.45, "positive": 0.05}, min_top_prob=0.4, min_margin=0.2)


def test_normalize_symbol_roundtrip() -> None:
    assert normalize_symbol("aapl") == "AAPL"
    assert normalize_symbol("  msft ") == "MSFT"
    assert normalize_symbol("") is None
    assert normalize_symbol("TOO_LONG_TICKER") is None


def test_s3io_write_and_read_jsonl() -> None:
    from finsense_shared import s3io

    fake = _FakeS3()
    rows = [{"a": 1}, {"b": "two"}, {"nested": {"k": 3}}]
    n = s3io.write_jsonl("mybucket", "raw/dt=2026-04-23/symbol=AAPL/run-1.jsonl", rows, client=fake)
    assert n == 3
    stored = fake.objects[("mybucket", "raw/dt=2026-04-23/symbol=AAPL/run-1.jsonl")]
    body = stored["Body"]
    assert isinstance(body, bytes)
    assert body.decode("utf-8").splitlines() == [json.dumps(r, ensure_ascii=False) for r in rows]
    assert stored["ContentType"] == "application/x-ndjson"
    assert stored["ServerSideEncryption"] == "AES256"

    got = list(s3io.read_jsonl("mybucket", "raw/dt=2026-04-23/symbol=AAPL/run-1.jsonl", client=fake))
    assert got == rows


def test_pseudo_label_text_echo_provider_is_deterministic() -> None:
    a = pseudo_label_text("EPS beat expectations", provider="echo", seed=7)
    b = pseudo_label_text("EPS beat expectations", provider="echo", seed=7)
    assert a.label_id == b.label_id
    assert LABEL_ID_TO_STR[a.label_id] == a.label_name
    assert a.provider == "echo"


def test_normalize_label_accepts_sentences() -> None:
    assert normalize_label("the article tone is positive today") == 2
    assert normalize_label("NEGATIVE\noutlook") == 0
    assert normalize_label("mixed signals") is None
