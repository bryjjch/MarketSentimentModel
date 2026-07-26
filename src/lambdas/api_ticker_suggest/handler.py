"""GET /tickers/suggest — ticker autocomplete by prefix over the valid-ticker universe.

Touches no DynamoDB: the universe comes from SSM / env / the packaged JSON file via
``finsense_shared.tickers.universe``.
"""

from __future__ import annotations

import os
from typing import Any

from finsense_shared import search_tickers_by_prefix
from finsense_shared.http import parse_limit, query_params, response

_DEFAULT_SUGGEST_LIMIT = int(os.environ.get("DEFAULT_SUGGEST_LIMIT", "10"))
_MAX_SUGGEST_LIMIT = int(os.environ.get("MAX_SUGGEST_LIMIT", "25"))


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    query = query_params(event)
    q = (query.get("q") or "").strip()
    if not q:
        return response(400, {"error": "bad_request", "message": "q is required"})
    try:
        limit = parse_limit(query, default=_DEFAULT_SUGGEST_LIMIT, maximum=_MAX_SUGGEST_LIMIT)
    except ValueError as exc:
        return response(400, {"error": "bad_request", "message": str(exc)})
    suggestions = search_tickers_by_prefix(q, limit=limit)
    return response(200, {"query": q.upper(), "suggestions": suggestions})
