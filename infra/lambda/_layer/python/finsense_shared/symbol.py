"""Ticker symbol normalization (kept in sync with ``infra/lambda/sentiment/symbol.py``)."""

from __future__ import annotations

import re

_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")


def normalize_symbol(raw: str | None) -> str | None:
    """Uppercase strip; return ``None`` unless it matches ``_SYMBOL_RE`` (1-5 Latin letters)."""
    if raw is None:
        return None
    s = raw.strip().upper()
    if not s:
        return None
    if not _SYMBOL_RE.match(s):
        return None
    return s
