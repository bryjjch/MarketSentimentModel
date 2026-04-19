"""Fetch recent headlines via Google News RSS (no API key)."""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .base import CollectedItem

# Google News RSS search (English, US).
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def _fetch_rss(url: str, timeout_s: float = 8.0) -> str:
    """Download RSS XML with a stable User-Agent."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FinSense/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def collect_news_rss(symbol: str, *, max_items: int) -> list[CollectedItem]:
    """Return up to ``max_items`` news entries mentioning the ticker."""
    q = urllib.parse.quote_plus(f"{symbol} stock")
    url = f"{_GOOGLE_NEWS_RSS}?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        body = _fetch_rss(url)
    except OSError:
        return []

    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []

    # RSS 2.0: channel/item
    items: list[CollectedItem] = []
    for item in root.findall(".//item"):
        if len(items) >= max_items:
            break
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        desc = (desc_el.text or "").strip() if desc_el is not None else ""
        text = f"{title}\n{desc}".strip()[:2000]
        if not text:
            continue
        items.append(
            CollectedItem(
                title=title or symbol,
                url=link or url,
                text=text,
                source_type="news_rss",
            )
        )
    return items
