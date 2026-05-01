"""Load and query the valid ticker universe used by API validation/autocomplete."""

from __future__ import annotations

import json
import os
import time
from bisect import bisect_left
from pathlib import Path
from typing import Iterable

import boto3
from botocore.exceptions import ClientError

from .symbol import normalize_symbol
from .tickers import load_tickers

_DEFAULT_CACHE_TTL_SECONDS = 900
_DEFAULT_PREFIX_LIMIT = 10
_MAX_PREFIX_LIMIT = 100

_CACHE_AT: float | None = None
_CACHE_LIST: tuple[str, ...] = ()
_CACHE_SET: frozenset[str] = frozenset()
_CACHE_FINGERPRINT: tuple[str, str, str] | None = None


def _clean(seq: Iterable[object]) -> tuple[str, ...]:
    """Clean and normalize a sequence of ticker symbols."""
    seen: set[str] = set()
    out: list[str] = []
    for value in seq:
        sym = normalize_symbol(str(value) if value is not None else None)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    out.sort()
    return tuple(out)


def _parse_serialized_symbols(raw: str | None) -> tuple[str, ...]:
    """Parse and clean a serialized list of ticker symbols."""
    if not raw:
        return ()
    s = raw.strip()
    if not s:
        return ()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return _clean(parsed)
    except json.JSONDecodeError:
        pass
    # Accept newline/comma-separated fallbacks for operator convenience.
    tokens = [part.strip() for part in s.replace("\n", ",").split(",")]
    return _clean([t for t in tokens if t])


def _read_file_symbols(path_value: str | None) -> tuple[str, ...]:
    """Read and parse ticker symbols from a file."""
    if not path_value:
        return ()
    p = Path(path_value)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    try:
        content = p.read_text(encoding="utf-8")
    except OSError:
        return ()
    return _parse_serialized_symbols(content)


def _load_from_ssm(param_name: str | None, ssm_client: object | None) -> tuple[str, ...]:
    """Load ticker symbols from SSM."""
    if not param_name:
        return ()
    name = param_name.strip()
    if not name:
        return ()
    client = ssm_client or boto3.client("ssm")
    try:
        resp = client.get_parameter(Name=name)
    except ClientError:
        return ()
    raw = (resp.get("Parameter") or {}).get("Value")
    return _parse_serialized_symbols(raw if isinstance(raw, str) else None)


def _cache_ttl_seconds() -> int:
    """Get the cache TTL from the environment."""
    raw = (os.environ.get("VALID_TICKERS_CACHE_TTL_SECONDS") or "").strip()
    try:
        parsed = int(raw) if raw else _DEFAULT_CACHE_TTL_SECONDS
    except ValueError:
        parsed = _DEFAULT_CACHE_TTL_SECONDS
    return max(0, parsed)


def load_valid_tickers(
    *,
    force_reload: bool = False,
    ssm_client: object | None = None,
) -> tuple[str, ...]:
    """Return sorted unique tickers from SSM/env/file, with legacy fallback."""
    global _CACHE_AT, _CACHE_LIST, _CACHE_SET, _CACHE_FINGERPRINT

    now = time.time()
    ttl = _cache_ttl_seconds()
    fingerprint = (
        (os.environ.get("VALID_TICKERS_SSM_PARAM") or "").strip(),
        (os.environ.get("VALID_TICKERS_JSON") or "").strip(),
        (os.environ.get("VALID_TICKERS_FILE") or "").strip(),
    )
    if (
        not force_reload
        and _CACHE_FINGERPRINT == fingerprint
        and _CACHE_AT is not None
        and (ttl == 0 or (now - _CACHE_AT) <= ttl)
        and _CACHE_LIST
    ):
        return _CACHE_LIST

    symbols = _load_from_ssm(os.environ.get("VALID_TICKERS_SSM_PARAM"), ssm_client)
    if not symbols:
        symbols = _parse_serialized_symbols(os.environ.get("VALID_TICKERS_JSON"))
    if not symbols:
        symbols = _read_file_symbols(os.environ.get("VALID_TICKERS_FILE"))
    if not symbols:
        symbols = _clean(load_tickers())

    _CACHE_LIST = symbols
    _CACHE_SET = frozenset(symbols)
    _CACHE_AT = now
    _CACHE_FINGERPRINT = fingerprint
    return _CACHE_LIST


def load_valid_ticker_set(*, force_reload: bool = False, ssm_client: object | None = None) -> frozenset[str]:
    """Return ticker membership set for O(1) checks."""
    if force_reload or not _CACHE_SET:
        load_valid_tickers(force_reload=force_reload, ssm_client=ssm_client)
    return _CACHE_SET


def search_tickers_by_prefix(prefix: str | None, *, limit: int = _DEFAULT_PREFIX_LIMIT) -> list[str]:
    """Return sorted prefix matches, capped by ``limit``."""
    normalized = normalize_symbol(prefix) if prefix is not None else None
    if not normalized:
        return []
    symbols = load_valid_tickers()
    capped = max(1, min(limit, _MAX_PREFIX_LIMIT))
    start = bisect_left(symbols, normalized)
    out: list[str] = []
    for i in range(start, len(symbols)):
        sym = symbols[i]
        if not sym.startswith(normalized):
            break
        out.append(sym)
        if len(out) >= capped:
            break
    return out
