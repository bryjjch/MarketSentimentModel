"""Lookup canonical company names for tickers (used to enrich news search and relevance)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .symbols import normalize_symbol
from .universe import DATA_DIR

logger = logging.getLogger(__name__)

_DEFAULT_FILENAME = "ticker_names_us.json"

_CACHE_FP: str | None = None
_CACHE_NAMES: dict[str, str] = {}


def _resolve_path() -> Path:
    """Return the JSON path; ``TICKER_NAMES_FILE`` overrides the bundled asset."""
    override = os.environ.get("TICKER_NAMES_FILE", "").strip()
    if override:
        return Path(override)
    return DATA_DIR / _DEFAULT_FILENAME


def _fingerprint() -> str:
    return os.environ.get("TICKER_NAMES_FILE", "").strip()


def _load_names() -> dict[str, str]:
    """Read the JSON map; returns empty dict on missing file or parse error."""
    path = _resolve_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("ticker_names_invalid_json %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in data.items():
        sym = normalize_symbol(str(key) if key is not None else None)
        if not sym:
            continue
        if not isinstance(value, str):
            continue
        name = value.strip()
        if not name:
            continue
        out[sym] = name
    return out


def _names_cached() -> dict[str, str]:
    global _CACHE_FP, _CACHE_NAMES
    fp = _fingerprint()
    if _CACHE_NAMES and fp == _CACHE_FP:
        return _CACHE_NAMES
    _CACHE_FP = fp
    _CACHE_NAMES = _load_names()
    return _CACHE_NAMES


def get_company_name(symbol: str | None) -> str | None:
    """Return canonical company name for ``symbol`` or ``None`` when unknown."""
    sym = normalize_symbol(symbol) if symbol is not None else None
    if not sym:
        return None
    return _names_cached().get(sym)


def clear_ticker_names_cache_for_tests() -> None:
    """Reset the module cache (tests only)."""
    global _CACHE_FP, _CACHE_NAMES
    _CACHE_FP = None
    _CACHE_NAMES = {}
