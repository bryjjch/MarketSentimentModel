"""The ticker universe: the daily run list, and the set the API will accept.

Two related lookups, both resolved from SSM with env/file fallbacks:

* :func:`load_tickers` — symbols the daily pipeline fans out over (``pipeline_dispatch``).
* :func:`load_valid_tickers` — everything the API accepts and autocompletes. Cached
  in-process behind a TTL because ``api_ticker_suggest`` hits it on every keystroke.

The bundled ``data/valid_tickers_us.json`` is not read unless ``VALID_TICKERS_FILE``
points at it; a bare filename in that variable resolves against ``data/``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from bisect import bisect_left
from pathlib import Path
from typing import Iterable

import boto3
from botocore.exceptions import ClientError

from .symbols import normalize_symbol

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"

_DEFAULT_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL")
_DEFAULT_CACHE_TTL_SECONDS = 900
_DEFAULT_PREFIX_LIMIT = 10
_MAX_PREFIX_LIMIT = 100

_CACHE_AT: float | None = None
_CACHE_LIST: tuple[str, ...] = ()
_CACHE_SET: frozenset[str] = frozenset()
_CACHE_FINGERPRINT: tuple[str, str, str] | None = None


def _read_ssm(param_name: str | None, ssm_client: object | None) -> str | None:
    """Return the raw SSM parameter value, or ``None`` when unset or unreadable."""
    name = (param_name or "").strip()
    if not name:
        return None
    client = ssm_client or boto3.client("ssm")
    try:
        resp = client.get_parameter(Name=name)
    except ClientError as e:
        logger.warning("ssm_read_failed param=%s: %s", name, e)
        return None
    value = (resp.get("Parameter") or {}).get("Value")
    return value if isinstance(value, str) else None


def _upper(seq: Iterable[object]) -> list[str]:
    """Uppercase non-empty entries in order, without enforcing symbol syntax."""
    out: list[str] = []
    for s in seq:
        if s is None:
            continue
        v = str(s).strip().upper()
        if v:
            out.append(v)
    return out


def _symbols(seq: Iterable[object]) -> tuple[str, ...]:
    """Normalize to valid symbols, de-duplicated and sorted."""
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
    """Parse a JSON list of symbols, falling back to comma/newline separated text."""
    s = (raw or "").strip()
    if not s:
        return ()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return _symbols(parsed)
    except json.JSONDecodeError:
        pass
    # Accept newline/comma-separated fallbacks for operator convenience.
    tokens = [part.strip() for part in s.replace("\n", ",").split(",")]
    return _symbols([t for t in tokens if t])


def _read_file_symbols(path_value: str | None) -> tuple[str, ...]:
    """Read symbols from a file; relative paths resolve against the bundled ``data/``."""
    if not path_value:
        return ()
    p = Path(path_value)
    if not p.is_absolute():
        p = DATA_DIR / p
    try:
        content = p.read_text(encoding="utf-8")
    except OSError:
        return ()
    return _parse_serialized_symbols(content)


def load_tickers(
    *,
    ssm_param: str | None = None,
    default_json: str | None = None,
    ssm_client: object | None = None,
) -> list[str]:
    """Load tickers from SSM JSON if configured, else from ``default_json``, else from built-ins."""
    raw = _read_ssm(ssm_param or os.environ.get("TOP_TICKERS_SSM_PARAM"), ssm_client)
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("ssm_tickers_invalid_json: %s", e)
            data = None
        if isinstance(data, list):
            values = _upper(data)
            if values:
                return values

    default_json = default_json if default_json is not None else os.environ.get("DEFAULT_TICKERS_JSON")
    if default_json:
        try:
            data = json.loads(default_json)
            if isinstance(data, list):
                values = _upper(data)
                if values:
                    return values
        except json.JSONDecodeError:
            pass
    return list(_DEFAULT_TICKERS)


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
    """Return sorted unique tickers from SSM/env/file, falling back to ``load_tickers``."""
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

    symbols = _parse_serialized_symbols(_read_ssm(os.environ.get("VALID_TICKERS_SSM_PARAM"), ssm_client))
    if not symbols:
        symbols = _parse_serialized_symbols(os.environ.get("VALID_TICKERS_JSON"))
    if not symbols:
        symbols = _read_file_symbols(os.environ.get("VALID_TICKERS_FILE"))
    if not symbols:
        symbols = _symbols(load_tickers())

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
