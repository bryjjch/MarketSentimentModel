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
    """Load the secret JSON from the ARN."""
    # Get the ARN from the environment variable
    arn = os.environ.get("REDDIT_SECRET_ARN", "").strip()
    if not arn:
        return None
    try:
        import boto3

        client = boto3.client("secretsmanager")
        # Get the secret value
        resp = client.get_secret_value(SecretId=arn)
        # Get the secret string
        raw = resp.get("SecretString") or ""
        # Parse the secret string as JSON
        return json.loads(raw)
    except Exception:
        return None


def _get_app_token(client_id: str, client_secret: str) -> str | None:
    """Get the app token from the Reddit API."""
    # Create the data
    data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
    # Create the request
    req = urllib.request.Request(
        _TOKEN_URL,
        data=data,
        headers={
            "User-Agent": "FinSenseSentimentBot/1.0",
        },
        method="POST",
    )
    # Create the base64 encoded string
    b64 = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    # Add the authorization header
    req.add_header("Authorization", f"Basic {b64}")

    # Try to get the app token
    try:
        # Open the request
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload.get("access_token", "")) or None
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        return None


def _search_posts(token: str, symbol: str, limit: int) -> list[dict[str, Any]]:
    """Search for posts from reddit"""
    # Create the query
    q = urllib.parse.urlencode(
        {
            "q": symbol,
            "limit": str(min(limit, 25)),
            "sort": "new",
            "type": "link",
            "restrict_sr": "false",
        }
    )
    # Create the URL
    url = f"{_SEARCH_URL}?{q}"
    # Create the request
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "FinSenseSentimentBot/1.0",
        },
    )
    # Try to get the data
    try:
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        return []

    out: list[dict[str, Any]] = []
    # Get the listing
    listing = data.get("data", {}).get("children", [])
    # For each child in the listing
    for child in listing:
        # Get the data
        cdata = child.get("data") or {}
        title = (cdata.get("title") or "").strip()
        permalink = cdata.get("permalink") or ""
        url_link = f"https://www.reddit.com{permalink}" if permalink else ""
        selftext = (cdata.get("selftext") or "").strip()[:1500]
        # Check if the title is empty
        if not title:
            # Continue to the next child
            continue
        # Get the text
        text = f"{title}\n{selftext}".strip()[:2000]
        # Add the item to the list
        out.append({"title": title, "url": url_link or "https://www.reddit.com", "text": text})
    return out


def collect_reddit(symbol: str, *, max_items: int) -> list[CollectedItem]:
    """Return Reddit posts if credentials are configured; otherwise []."""
    # Load the secret JSON
    secret = _load_secret_json()
    # Check if the secret is None
    if not secret:
        return []

    # Get the client ID and client secret
    client_id = str(secret.get("client_id", "") or "").strip()
    # Check if the client secret is empty
    client_secret = str(secret.get("client_secret", "") or "").strip()
    # Check if the client ID or client secret is empty
    if not client_id or not client_secret:
        return []

    # Get the app token
    token = _get_app_token(client_id, client_secret)
    # Check if the token is None
    if not token:
        return []

    # Search for posts
    posts = _search_posts(token, symbol, max_items)
    # Create the list of items
    items: list[CollectedItem] = []
    for p in posts[:max_items]:
        # Add the item to the list
        items.append(
            CollectedItem(
                title=p["title"],
                url=p["url"],
                text=p["text"],
                source_type="reddit",
            )
        )
    return items
