"""News/social source adapters (shared between on-demand sentiment Lambda and daily ingestion)."""

from __future__ import annotations

from .base import CollectedItem
from .news_rss import collect_news_rss
from .reddit import collect_reddit


def collect_for_symbol(
    symbol: str,
    *,
    max_articles: int,
    include_social: bool,
) -> list[CollectedItem]:
    """Gather text items: Google News RSS, then Reddit (when credentials exist).

    Total item count is capped at ``max_articles`` (hard limit 40). Duplicates are
    removed by URL (falling back to title).
    """
    max_articles = max(1, min(max_articles, 40))

    if include_social:
        n_news = max(1, max_articles // 2)
        n_social = max_articles - n_news
    else:
        n_news = max_articles
        n_social = 0

    items: list[CollectedItem] = []
    items.extend(collect_news_rss(symbol, max_items=n_news))
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


__all__ = ["CollectedItem", "collect_for_symbol", "collect_news_rss", "collect_reddit"]
