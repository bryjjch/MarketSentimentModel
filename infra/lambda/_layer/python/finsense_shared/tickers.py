"""Resolve the daily ticker list from SSM (with env-var fallback)."""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_DEFAULT_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL")


def _clean(seq: Iterable[object]) -> list[str]:
    out: list[str] = []
    for s in seq:
        if s is None:
            continue
        v = str(s).strip().upper()
        if v:
            out.append(v)
    return out


def load_tickers(
    *,
    ssm_param: str | None = None,
    default_json: str | None = None,
    ssm_client: object | None = None,
) -> list[str]:
    """Load tickers from SSM JSON if configured, else from ``default_json``, else from built-ins."""
    ssm_param = (ssm_param or os.environ.get("TOP_TICKERS_SSM_PARAM", "")).strip()
    if ssm_param:
        client = ssm_client or boto3.client("ssm")
        try:
            resp = client.get_parameter(Name=ssm_param)
            raw = resp["Parameter"]["Value"]
            data = json.loads(raw)
            if isinstance(data, list):
                values = _clean(data)
                if values:
                    return values
        except (ClientError, json.JSONDecodeError, KeyError) as e:
            logger.warning("ssm_tickers_failed: %s", e)

    default_json = default_json if default_json is not None else os.environ.get("DEFAULT_TICKERS_JSON")
    if default_json:
        try:
            data = json.loads(default_json)
            if isinstance(data, list):
                values = _clean(data)
                if values:
                    return values
        except json.JSONDecodeError:
            pass
    return list(_DEFAULT_TICKERS)
