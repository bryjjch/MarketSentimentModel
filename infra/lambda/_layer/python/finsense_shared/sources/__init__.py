"""News/social source adapters (shared between on-demand sentiment Lambda and daily ingestion)."""

from __future__ import annotations

import os

from .base import CollectedItem
from .finnhub_news import collect_finnhub_news, finnhub_news_enabled
from .news_rss import collect_news_rss
from .reddit import collect_reddit


def _finnhub_fallback_rss_enabled() -> bool:
    return os.environ.get("FINNHUB_FALLBACK_RSS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _collect_news_slot(symbol: str, n_news: int) -> list[CollectedItem]:
    """Prefer Finnhub symbol-keyed company news when configured; optionally backfill with RSS."""
    if not finnhub_news_enabled():
        return collect_news_rss(symbol, max_items=n_news)

    finnhub_items = collect_finnhub_news(symbol, max_items=n_news)
    if len(finnhub_items) >= n_news:
        return finnhub_items[:n_news]

    short = n_news - len(finnhub_items)
    if short > 0 and _finnhub_fallback_rss_enabled():
        rss = collect_news_rss(symbol, max_items=short)
        seen_urls = {it.url for it in finnhub_items}
        merged = list(finnhub_items)
        for it in rss:
            if it.url in seen_urls:
                continue
            merged.append(it)
            seen_urls.add(it.url)
            if len(merged) >= n_news:
                break
        return merged[:n_news]
    return finnhub_items


def collect_for_symbol(
    symbol: str,
    *,
    max_articles: int,
    include_social: bool,
) -> list[CollectedItem]:
    """Gather text items: Finnhub (when configured) or Google News RSS, then Reddit if enabled.

    Total item count is capped at ``max_articles`` (hard limit 40). Duplicates are
    removed by URL (falling back to title).

    When Finnhub is configured but returns fewer than the news quota, RSS backfill
    runs only if ``FINNHUB_FALLBACK_RSS`` is ``true`` (default off for accuracy).
    """
    max_articles = max(1, min(max_articles, 40))

    if include_social:
        n_news = max(1, max_articles // 2)
        n_social = max_articles - n_news
    else:
        n_news = max_articles
        n_social = 0

    items: list[CollectedItem] = []
    items.extend(_collect_news_slot(symbol, n_news))
    if n_social > 0:
        items.extend(collect_reddit(symbol, max_items=n_social))

    seen: set[str] = set()
    unique: list[CollectedItem] = []
    for it in items:
        key = it.url or it.title
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
        if len(unique) >= max_articles:
            break
    return unique


__all__ = [
    "CollectedItem",
    "collect_for_symbol",
    "collect_finnhub_news",
    "finnhub_news_enabled",
    "collect_news_rss",
    "collect_reddit",
]
