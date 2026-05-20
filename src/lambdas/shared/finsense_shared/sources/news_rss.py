"""Fetch recent headlines via Google News RSS (no API key) with rules-based relevance filtering."""

from __future__ import annotations

import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from ..ticker_names import get_company_name
from ..valid_tickers import load_valid_ticker_set
from .base import CollectedItem

logger = logging.getLogger(__name__)

# Google News RSS search (English, US).
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

# Headlines that are clearly broad-market/index wraps. Drop unless company name is also present.
_MACRO_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bS&P\s?500\b", re.IGNORECASE),
    re.compile(r"\bnasdaq( composite)?\b", re.IGNORECASE),
    re.compile(r"\bdow( jones)?\b", re.IGNORECASE),
    re.compile(r"\bwall street\b", re.IGNORECASE),
    re.compile(r"\bmarkets?\s+(open|close|wrap|recap|today)\b", re.IGNORECASE),
)

# Token that "looks like" a ticker (1-5 uppercase letters), used to detect watchlist/round-up titles.
_TICKER_LIKE_RE = re.compile(r"\b[A-Z]{1,5}\b")
_MULTI_TICKER_THRESHOLD = 3

# Tunable over-fetch multiplier; capped to keep RSS calls bounded.
_DEFAULT_OVERFETCH = 3
_OVERFETCH_HARD_CAP = 60


def _overfetch_multiplier() -> int:
    raw = (os.environ.get("RSS_OVERFETCH") or "").strip()
    try:
        n = int(raw) if raw else _DEFAULT_OVERFETCH
    except ValueError:
        n = _DEFAULT_OVERFETCH
    return max(1, n)


def _build_query(symbol: str, name: str | None) -> str:
    """Return the Google News query string for ``symbol`` (and optional company ``name``)."""
    sym = symbol.upper()
    if name:
        return f'"{name}" OR "{sym}"'
    return f'"{sym}" stock'


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


def _has_whole_word(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    pattern = re.compile(rf"\b{re.escape(needle)}\b", re.IGNORECASE)
    return bool(pattern.search(haystack))


def _name_match(haystack: str, name: str) -> bool:
    """Match either the full canonical name or its first token (when long enough)."""
    if _has_whole_word(haystack, name):
        return True
    first = name.split()[0] if name else ""
    if len(first) >= 4 and _has_whole_word(haystack, first):
        return True
    return False


def _looks_macro(title: str, name: str | None) -> bool:
    """True when title clearly matches an index/market-wrap pattern and lacks the company name."""
    if not any(p.search(title) for p in _MACRO_PATTERNS):
        return False
    if name and _name_match(title, name):
        return False
    return True


def _multi_ticker_title(title: str, symbol: str) -> bool:
    """Drop watch-list titles whose headline lists ~3+ distinct known tickers."""
    candidates = {tok.upper() for tok in _TICKER_LIKE_RE.findall(title)}
    if not candidates:
        return False
    try:
        valid = load_valid_ticker_set()
    except Exception:  # noqa: BLE001 -- never let lookup failures block filtering
        return False
    distinct_tickers = {t for t in candidates if t in valid and t != symbol.upper()}
    # +1 to account for the requested symbol itself if it appeared in the title.
    total = len(distinct_tickers) + (1 if symbol.upper() in candidates else 0)
    return total >= _MULTI_TICKER_THRESHOLD


def _is_relevant(title: str, description: str, symbol: str, name: str | None) -> bool:
    """Return True when title/description is clearly about ``symbol``/``name``."""
    haystack = f"{title}\n{description}"
    if _looks_macro(title, name):
        return False
    if _multi_ticker_title(title, symbol):
        return False
    if _has_whole_word(haystack, symbol):
        return True
    if name and _name_match(haystack, name):
        return True
    return False


def collect_news_rss(symbol: str, *, max_items: int) -> list[CollectedItem]:
    """Return up to ``max_items`` news entries clearly about the ticker.

    Over-fetches RSS results then applies rules-only relevance filters to drop
    homonyms (e.g. "costs" for COST), broad market/index wraps, and watch-list
    headlines that mention many tickers.
    """
    if max_items <= 0:
        return []
    name = get_company_name(symbol)
    overfetch = _overfetch_multiplier()
    fetch_target = min(max_items * overfetch, _OVERFETCH_HARD_CAP)

    q = urllib.parse.quote_plus(_build_query(symbol, name))
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
    inspected = 0
    for item in root.findall(".//item"):
        if len(items) >= max_items or inspected >= fetch_target:
            break
        inspected += 1
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        desc = (desc_el.text or "").strip() if desc_el is not None else ""
        text = f"{title}\n{desc}".strip()[:2000]
        if not text:
            continue
        if not _is_relevant(title, desc, symbol, name):
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
