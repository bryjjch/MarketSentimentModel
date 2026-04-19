"""Ticker symbol normalization and light validation."""

from __future__ import annotations

import re

# Typical US equity ticker: 1-5 letters (allow common formats; extend as needed)
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
