"""Tests for Finnhub symbol-keyed news and news-slot routing."""

from __future__ import annotations

import json

import boto3
import pytest

from finsense_shared.sources.base import CollectedItem
from finsense_shared.sources.finnhub_news import (
    clear_finnhub_api_key_cache_for_tests,
    collect_finnhub_news,
    finnhub_news_enabled,
    get_finnhub_api_key,
)


def test_finnhub_news_enabled_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_finnhub_api_key_cache_for_tests()
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_SECRET_ARN", raising=False)
    assert finnhub_news_enabled() is False


def test_finnhub_api_key_secrets_manager_called_once(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_finnhub_api_key_cache_for_tests()
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("FINNHUB_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:0:secret:test")
    calls: list[int] = []

    class FakeSM:
        def get_secret_value(self, **kwargs: object) -> dict[str, str]:
            calls.append(1)
            return {"SecretString": '{"api_key":"one-read"}'}

    def fake_boto_client(name: str, **kwargs: object) -> FakeSM:
        assert name == "secretsmanager"
        return FakeSM()

    monkeypatch.setattr(boto3, "client", fake_boto_client)

    assert get_finnhub_api_key() == "one-read"
    assert get_finnhub_api_key() == "one-read"
    assert finnhub_news_enabled() is True
    assert len(calls) == 1


def test_collect_finnhub_filters_by_related_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "test-token")
    payload = [
        {
            "headline": "Older story",
            "summary": "Plain summary",
            "url": "https://example.com/old",
            "related": "COST",
            "datetime": 1000,
            "source": "X",
        },
        {
            "headline": "Wrong ticker",
            "summary": "",
            "url": "https://example.com/wrong",
            "related": "META",
            "datetime": 9999999999999,
            "source": "Y",
        },
        {
            "headline": "Newer story",
            "summary": "<p>Hi&nbsp;there</p>",
            "url": "https://example.com/new",
            "related": "COST,META",
            "datetime": 2000,
            "source": "Z",
        },
    ]

    class FakeResp:
        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "finsense_shared.sources.finnhub_news.urllib.request.urlopen",
        lambda req, timeout=12.0: FakeResp(),
    )

    items = collect_finnhub_news("COST", max_items=10)
    assert len(items) == 2
    assert items[0].title == "Newer story"
    assert "Hi there" in items[0].text or "there" in items[0].text
    assert items[0].source_type == "finnhub"
    assert items[1].title == "Older story"


def test_collect_for_symbol_skips_rss_when_finnhub_fills_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "x")

    def fake_finnhub(symbol: str, *, max_items: int, api_key: str | None = None) -> list[CollectedItem]:
        return [
            CollectedItem(
                title="t",
                url=f"https://z/{symbol}/{i}",
                text="body",
                source_type="finnhub",
            )
            for i in range(max_items)
        ]

    def fake_rss(*_a: object, **_k: object) -> list[CollectedItem]:
        raise AssertionError("RSS should not run when Finnhub fills the news quota")

    monkeypatch.setattr("finsense_shared.sources.collect_finnhub_news", fake_finnhub)
    monkeypatch.setattr("finsense_shared.sources.collect_news_rss", fake_rss)

    from finsense_shared.sources import collect_for_symbol

    out = collect_for_symbol("AAPL", max_articles=6)
    assert len(out) == 6
    assert all(it.source_type == "finnhub" for it in out)


def test_collect_for_symbol_falls_back_to_rss_without_finnhub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_SECRET_ARN", raising=False)

    def fake_rss(symbol: str, *, max_items: int) -> list[CollectedItem]:
        return [
            CollectedItem(
                title="rss",
                url="https://rss",
                text="x",
                source_type="news_rss",
            )
        ]

    monkeypatch.setattr("finsense_shared.sources.collect_news_rss", fake_rss)

    from finsense_shared.sources import collect_for_symbol

    out = collect_for_symbol("AAPL", max_articles=4)
    assert len(out) == 1
    assert out[0].source_type == "news_rss"
