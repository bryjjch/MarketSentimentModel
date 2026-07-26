"""Ticker symbols: syntax, the universe we run over, and company names.

* ``symbols`` — :func:`normalize_symbol`, the one definition of a well-formed ticker.
* ``universe`` — which symbols the pipeline runs and the API accepts.
* ``names`` — canonical company names, used to enrich news search.
* ``data/`` — bundled US ticker and company-name JSON.
"""

from __future__ import annotations

from .names import get_company_name
from .symbols import normalize_symbol
from .universe import (
    load_tickers,
    load_valid_ticker_set,
    load_valid_tickers,
    search_tickers_by_prefix,
)

__all__ = [
    "get_company_name",
    "load_tickers",
    "load_valid_ticker_set",
    "load_valid_tickers",
    "normalize_symbol",
    "search_tickers_by_prefix",
]
