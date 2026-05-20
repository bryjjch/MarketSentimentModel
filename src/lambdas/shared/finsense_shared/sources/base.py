"""Source adapters: fetch candidate texts for sentiment (news, social, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class CollectedItem:
    """One article/post title, link, text for inference, and origin tag."""

    title: str
    url: str
    text: str
    source_type: str  # e.g. "news_rss", "reddit"


class SourceAdapter(Protocol):
    """Pluggable source: returns items for a normalized ticker symbol."""

    def collect(self, symbol: str, *, max_items: int, include_social: bool) -> list[CollectedItem]:
        ...
