"""Tests for ticker name lookup, RSS query construction, relevance filter, and over-fetch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finsense_shared import sources as sources_pkg
from finsense_shared import ticker_names
from finsense_shared.sources import news_rss


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a fresh ticker-name cache and the bundled JSON file."""
    ticker_names.clear_ticker_names_cache_for_tests()
    monkeypatch.delenv("TICKER_NAMES_FILE", raising=False)
    monkeypatch.delenv("RSS_OVERFETCH", raising=False)


# ---------------------------------------------------------------------------
# ticker_names loader


def test_get_company_name_for_known_symbol() -> None:
    assert ticker_names.get_company_name("AAPL") == "Apple"
    assert ticker_names.get_company_name("aapl") == "Apple"


def test_get_company_name_unknown_returns_none() -> None:
    assert ticker_names.get_company_name("ZZZZZ") is None


def test_get_company_name_invalid_input_returns_none() -> None:
    assert ticker_names.get_company_name("") is None
    assert ticker_names.get_company_name(None) is None


def test_get_company_name_override_with_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TICKER_NAMES_FILE", str(tmp_path / "nope.json"))
    ticker_names.clear_ticker_names_cache_for_tests()
    assert ticker_names.get_company_name("AAPL") is None


def test_get_company_name_override_with_custom_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = tmp_path / "names.json"
    custom.write_text(json.dumps({"FOO": "Foo Industries", "BAR": ""}), encoding="utf-8")
    monkeypatch.setenv("TICKER_NAMES_FILE", str(custom))
    ticker_names.clear_ticker_names_cache_for_tests()
    assert ticker_names.get_company_name("FOO") == "Foo Industries"
    assert ticker_names.get_company_name("BAR") is None
    assert ticker_names.get_company_name("AAPL") is None


# ---------------------------------------------------------------------------
# _build_query


def test_build_query_with_known_name() -> None:
    assert news_rss._build_query("AAPL", "Apple") == '"Apple" OR "AAPL"'


def test_build_query_without_name_falls_back_to_ticker_stock() -> None:
    assert news_rss._build_query("ZZZZZ", None) == '"ZZZZZ" stock'


# ---------------------------------------------------------------------------
# _is_relevant


@pytest.mark.parametrize(
    "title,desc,symbol,name",
    [
        ("AAPL hits new high after earnings", "", "AAPL", "Apple"),
        ("Apple beats guidance", "Strong iPhone sales drove growth.", "AAPL", "Apple"),
        ("Costco quarterly comps surprise", "", "COST", "Costco"),
    ],
)
def test_is_relevant_keeps_clearly_relevant_titles(
    title: str,
    desc: str,
    symbol: str,
    name: str | None,
) -> None:
    assert news_rss._is_relevant(title, desc, symbol, name) is True


@pytest.mark.parametrize(
    "title,desc,symbol,name",
    [
        # Homonym noise: "costs" should not match COST.
        ("Costs rising for Meta as ad spend dips", "", "COST", "Costco"),
        # Macro / index wrap with no company name.
        ("Nasdaq, S&P 500 Log Record Finish, Extend Streak of Weekly Gains", "", "AAPL", "Apple"),
        # Stocks rise headline (broad market).
        ("Stocks rise as Fed signals pause", "", "MSFT", "Microsoft"),
    ],
)
def test_is_relevant_drops_irrelevant_titles(
    title: str,
    desc: str,
    symbol: str,
    name: str | None,
) -> None:
    assert news_rss._is_relevant(title, desc, symbol, name) is False


def test_is_relevant_drops_multi_ticker_watchlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        news_rss,
        "load_valid_ticker_set",
        lambda: frozenset({"AAPL", "MSFT", "NVDA", "GOOGL"}),
    )
    assert (
        news_rss._is_relevant("Top picks today: AAPL, MSFT, NVDA, GOOGL", "", "AAPL", "Apple")
        is False
    )


def test_is_relevant_macro_with_company_name_kept() -> None:
    assert (
        news_rss._is_relevant(
            "Apple drags on Nasdaq as record streak ends",
            "",
            "AAPL",
            "Apple",
        )
        is True
    )


# ---------------------------------------------------------------------------
# collect_news_rss + over-fetch


def _rss_xml(items: list[tuple[str, str, str]]) -> bytes:
    from xml.sax.saxutils import escape

    body = ['<?xml version="1.0"?><rss><channel>']
    for title, link, desc in items:
        body.append(
            f"<item><title>{escape(title)}</title><link>{escape(link)}</link>"
            f"<description>{escape(desc)}</description></item>"
        )
    body.append("</channel></rss>")
    return "".join(body).encode("utf-8")


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_collect_news_rss_overfetches_then_filters_and_trims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        ("Costs rising for rivals", "https://example.com/1", "macro"),  # drop
        ("Nasdaq, S&P 500 hit record finish", "https://example.com/2", "macro"),  # drop
        ("Costco posts strong comps", "https://example.com/3", ""),  # keep
        ("COST beats earnings", "https://example.com/4", ""),  # keep
        ("Costco expands warehouses", "https://example.com/5", ""),  # keep (cap reached)
        ("Costco raises dividend", "https://example.com/6", ""),  # not visited (over cap)
    ]
    captured_urls: list[str] = []

    def fake_urlopen(req: object, timeout: float = 8.0) -> _FakeResp:
        captured_urls.append(getattr(req, "full_url", ""))
        return _FakeResp(_rss_xml(items))

    monkeypatch.setattr(news_rss.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("RSS_OVERFETCH", "5")

    out = news_rss.collect_news_rss("COST", max_items=3)
    titles = [it.title for it in out]
    assert titles == [
        "Costco posts strong comps",
        "COST beats earnings",
        "Costco expands warehouses",
    ]
    assert all(it.source_type == "news_rss" for it in out)
    # The query should be Costco-OR-ticker since COST is in the bundled name map.
    assert "Costco" in captured_urls[0]


def test_collect_news_rss_returns_empty_when_all_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        ("Stocks rise on Fed pause", "https://example.com/a", ""),
        ("Costs surge across sector", "https://example.com/b", ""),
    ]
    monkeypatch.setattr(
        news_rss.urllib.request,
        "urlopen",
        lambda req, timeout=8.0: _FakeResp(_rss_xml(items)),
    )
    assert news_rss.collect_news_rss("COST", max_items=5) == []


def test_collect_news_rss_unknown_symbol_uses_ticker_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_urls: list[str] = []

    def fake_urlopen(req: object, timeout: float = 8.0) -> _FakeResp:
        captured_urls.append(getattr(req, "full_url", ""))
        return _FakeResp(_rss_xml([("ZZZZZ moves higher", "https://example.com/u", "")]))

    monkeypatch.setattr(news_rss.urllib.request, "urlopen", fake_urlopen)
    out = news_rss.collect_news_rss("ZZZZZ", max_items=1)
    assert len(out) == 1
    # ZZZZZ has no company name -> ticker-only query path.
    assert "ZZZZZ" in captured_urls[0]
    assert "stock" in captured_urls[0].lower()


# ---------------------------------------------------------------------------
# Public API still wired through collect_for_symbol


def test_collect_for_symbol_uses_filtered_news_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = sources_pkg.CollectedItem(
        title="Costco news",
        url="https://example.com/cost",
        text="Costco news",
        source_type="news_rss",
    )

    def fake_news(symbol: str, *, max_items: int) -> list:
        assert symbol == "COST"
        return [sentinel] * max_items

    monkeypatch.setattr(sources_pkg, "collect_news_rss", fake_news)
    monkeypatch.setattr(sources_pkg, "collect_reddit", lambda symbol, *, max_items: [])

    out = sources_pkg.collect_for_symbol("COST", max_articles=2, include_social=False)
    assert len(out) == 1  # de-duped by URL
    assert out[0].source_type == "news_rss"
