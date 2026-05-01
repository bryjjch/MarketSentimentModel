"""Tests for the shared ingestion/prediction/pseudo-label helpers (``finsense_shared``)."""

from __future__ import annotations

import io
import json
import sys
from types import ModuleType

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
    search_tickers_by_prefix,
)
import finsense_shared.valid_tickers as valid_tickers
from finsense_shared.confidence import ConfidenceMetric
from finsense_shared.llm_label import LABEL_ID_TO_STR, SYSTEM_PROMPT, normalize_label, pseudo_label_text


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


def test_valid_ticker_loader_from_json_and_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VALID_TICKERS_SSM_PARAM", "")
    monkeypatch.setenv("VALID_TICKERS_FILE", "")
    monkeypatch.setenv("VALID_TICKERS_JSON", "[\"aapl\", \"MSFT\", \"AAPL\", \"META\"]")
    valid_tickers.load_valid_tickers(force_reload=True)

    loaded = valid_tickers.load_valid_tickers()
    assert loaded == ("AAPL", "META", "MSFT")
    assert "AAPL" in valid_tickers.load_valid_ticker_set()
    assert search_tickers_by_prefix("aa", limit=5) == ["AAPL"]


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


def _install_fake_google_genai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: object,
) -> dict[str, object]:
    """Register minimal ``google`` / ``google.genai`` modules so ``_label_google`` can run without the real SDK."""

    last: dict[str, object] = {}

    class _FakeModels:
        def generate_content(self, *, model: str, contents: str, config: object) -> object:
            last["model"] = model
            last["contents"] = contents
            last["config"] = config
            return response

    class _FakeClient:
        def __init__(self, api_key: str | None = None) -> None:
            last["api_key"] = api_key
            self.models = _FakeModels()

    types_mod = ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(
            self,
            *,
            system_instruction: str | None = None,
            temperature: float | None = None,
        ) -> None:
            self.system_instruction = system_instruction
            self.temperature = temperature

    types_mod.GenerateContentConfig = GenerateContentConfig

    genai_mod = ModuleType("google.genai")
    genai_mod.Client = _FakeClient
    genai_mod.types = types_mod

    google_mod = ModuleType("google")
    google_mod.__path__ = []

    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google", google_mod)

    return last


def test_pseudo_label_text_google_provider_parses_resp_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    resp = ModuleType("resp")
    resp.text = "positive"
    resp.candidates = []

    last = _install_fake_google_genai(monkeypatch, response=resp)

    out = pseudo_label_text(
        "EPS beat expectations",
        provider="google",
        model="gemini-test",
        temperature=0.1,
    )

    assert out.provider == "google"
    assert out.label_id == 2
    assert out.label_name == "positive"
    assert out.model == "gemini-test"

    cfg = last["config"]
    assert getattr(cfg, "system_instruction", None) == SYSTEM_PROMPT
    assert getattr(cfg, "temperature", None) == pytest.approx(0.1)
    assert last["model"] == "gemini-test"
    assert last["api_key"] == "test-google-key"
    assert "EPS beat expectations" in str(last["contents"])


def test_pseudo_label_text_google_provider_fallback_to_candidates_when_text_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    class _Part:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Content:
        def __init__(self, parts: list[_Part]) -> None:
            self.parts = parts

    class _Candidate:
        def __init__(self, parts: list[_Part]) -> None:
            self.content = _Content(parts)

    resp = ModuleType("resp")
    resp.text = ""
    resp.candidates = [_Candidate([_Part("negative")])]

    _install_fake_google_genai(monkeypatch, response=resp)

    out = pseudo_label_text("Guidance cut", provider="google")
    assert out.label_id == 0
    assert out.label_name == "negative"


def test_normalize_label_accepts_sentences() -> None:
    assert normalize_label("the article tone is positive today") == 2
    assert normalize_label("NEGATIVE\noutlook") == 0
    assert normalize_label("mixed signals") is None
