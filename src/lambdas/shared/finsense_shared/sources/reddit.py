"""Optional Reddit collection via OAuth client-credentials (requires client id/secret in Secrets Manager)."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .base import CollectedItem

_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_SEARCH_URL = "https://oauth.reddit.com/search"


def _load_secret_json() -> dict[str, Any] | None:
    """Load ``client_id`` / ``client_secret`` from Secrets Manager when ``REDDIT_SECRET_ARN`` is set."""
    arn = os.environ.get("REDDIT_SECRET_ARN", "").strip()
    if not arn:
        return None
    try:
        import boto3

        client = boto3.client("secretsmanager")
        resp = client.get_secret_value(SecretId=arn)
        raw = resp.get("SecretString") or ""
        return json.loads(raw)
    except Exception:
        return None


def _get_app_token(client_id: str, client_secret: str) -> str | None:
    """Client-credentials access token for application-only OAuth."""
    data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
    req = urllib.request.Request(
        _TOKEN_URL,
        data=data,
        headers={
            "User-Agent": "FinSenseSentimentBot/1.0",
        },
        method="POST",
    )
    b64 = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    req.add_header("Authorization", f"Basic {b64}")

    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload.get("access_token", "")) or None
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        return None


def _search_posts(token: str, symbol: str, limit: int) -> list[dict[str, Any]]:
    """Search recent link posts; caps ``limit`` to Reddit's usual max per request."""
    q = urllib.parse.urlencode(
        {
            "q": symbol,
            "limit": str(min(limit, 25)),
            "sort": "new",
            "type": "link",
            "restrict_sr": "false",
        }
    )
    url = f"{_SEARCH_URL}?{q}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "FinSenseSentimentBot/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        return []

    out: list[dict[str, Any]] = []
    listing = data.get("data", {}).get("children", [])
    for child in listing:
        cdata = child.get("data") or {}
        title = (cdata.get("title") or "").strip()
        permalink = cdata.get("permalink") or ""
        url_link = f"https://www.reddit.com{permalink}" if permalink else ""
        selftext = (cdata.get("selftext") or "").strip()[:1500]
        if not title:
            continue
        text = f"{title}\n{selftext}".strip()[:2000]
        out.append({"title": title, "url": url_link or "https://www.reddit.com", "text": text})
    return out


def collect_reddit(symbol: str, *, max_items: int) -> list[CollectedItem]:
    """Return Reddit posts if credentials are configured; otherwise ``[]``."""
    secret = _load_secret_json()
    if not secret:
        return []

    client_id = str(secret.get("client_id", "") or "").strip()
    client_secret = str(secret.get("client_secret", "") or "").strip()
    if not client_id or not client_secret:
        return []

    token = _get_app_token(client_id, client_secret)
    if not token:
        return []

    posts = _search_posts(token, symbol, max_items)
    items: list[CollectedItem] = []
    for p in posts[:max_items]:
        items.append(
            CollectedItem(
                title=p["title"],
                url=p["url"],
                text=p["text"],
                source_type="reddit",
            )
        )
    return items
