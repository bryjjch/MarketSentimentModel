"""Provider-agnostic pseudo-labeler for 3-class sentiment.

Supports ``openai`` (Chat Completions), ``google`` (Gemini via Google AI Studio), and
``echo`` (offline deterministic stub for plumbing tests).

Secrets are resolved from AWS Secrets Manager when ``*_SECRET_ARN`` env vars are set;
otherwise from the matching ``*_API_KEY`` env var. This matches how other Lambdas in
the stack pick up credentials.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import boto3

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You label financial-news sentences for sentiment toward the underlying company "
    "or market event. Reply with exactly one token: negative, neutral, or positive."
)
USER_TEMPLATE = (
    'Classify sentiment (one word: negative, neutral, or positive):\n\n"""{text}"""'
)

LABEL_ID_TO_STR = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_STR_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


@dataclass(frozen=True)
class PseudoLabel:
    label_id: int
    label_name: str
    provider: str
    model: str


def normalize_label(raw: str) -> int | None:
    """Map model text to 0/1/2: prefer whole-word sentiment tokens, else stripped alphanumeric lookup."""
    low = (raw or "").strip().lower()
    for word, lid in (("negative", 0), ("neutral", 1), ("positive", 2)):
        if re.search(rf"\b{word}\b", low):
            return lid
    token = re.sub(r"[^a-z]+", "", low)
    return LABEL_STR_TO_ID.get(token)


def _load_secret_json(arn: str) -> dict[str, Any] | None:
    try:
        client = boto3.client("secretsmanager")
        resp = client.get_secret_value(SecretId=arn)
        raw = resp.get("SecretString") or ""
        data = json.loads(raw) if raw else None
    except Exception as e:  # noqa: BLE001 -- missing creds should never crash the pipeline
        logger.warning("secrets_manager_read_failed arn=%s err=%s", arn, e)
        return None
    return data if isinstance(data, dict) else None


def _resolve_api_key(provider: str) -> str | None:
    """Load an API key for ``provider`` from Secrets Manager (if ARN configured) or env."""
    arn = os.environ.get(f"{provider.upper()}_SECRET_ARN", "").strip()
    if arn:
        data = _load_secret_json(arn)
        if data:
            for k in ("api_key", "apiKey", "key", "token"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    if provider == "openai":
        return (os.environ.get("OPENAI_API_KEY") or "").strip() or None
    if provider == "google":
        return (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        ).strip() or None
    return None


def _http_post_json(url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 -- trusted API URL
        raw = resp.read().decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"non-JSON response from {url}: {raw[:200]!r}") from e


def _label_openai(text: str, model: str, temperature: float, timeout_s: float, api_key: str) -> int:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ],
    }
    data = _http_post_json(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        payload=payload,
        timeout_s=timeout_s,
    )
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"openai empty choices: {data}")
    content = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
    lid = normalize_label(content)
    if lid is None:
        raise ValueError(f"unparseable openai output: {content!r}")
    return lid


def _label_google(text: str, model: str, temperature: float, timeout_s: float, api_key: str) -> int:
    try:
        from google import genai  # pyright: ignore[reportMissingImports]
        from google.genai import types  # pyright: ignore[reportMissingImports]
    except ImportError as e:  # pragma: no cover - runtime dependency check
        raise RuntimeError("google provider requires dependency `google-genai`") from e

    client = genai.Client(api_key=api_key)
    user = USER_TEMPLATE.format(text=text)
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            # The SDK does not expose a simple per-request timeout in this call path.
            # Keep timeout_s in the signature for provider parity and future tuning.
        ),
    )
    content = (getattr(resp, "text", "") or "").strip()
    if not content:
        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
            raise ValueError(f"google empty candidates: {resp!r}")
        parts = ((getattr(getattr(candidates[0], "content", None), "parts", None)) or [])
        content = "".join(str(getattr(p, "text", "") or "") for p in parts).strip()
    lid = normalize_label(content)
    if lid is None:
        raise ValueError(f"unparseable google output: {content!r}")
    return lid


def _label_echo(text: str, seed: int) -> int:
    """Deterministic offline stub: hash text to one of {0,1,2}."""
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return rng.randint(0, 2)


def default_model(provider: str) -> str:
    if provider == "openai":
        return "gpt-4o-mini"
    if provider == "google":
        return "gemini-3.1-flash-lite"
    return "echo-stub"


def pseudo_label_text(
    text: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    timeout_s: float = 15.0,
    max_chars: int = 4000,
    seed: int = 42,
) -> PseudoLabel:
    """Label a single text with the selected provider and return a ``PseudoLabel``.

    Provider selection precedence: explicit arg > ``LLM_PROVIDER`` env var > ``openai``.
    """
    provider = (provider or os.environ.get("LLM_PROVIDER") or "openai").strip().lower()
    if provider not in ("openai", "google", "echo"):
        raise ValueError(f"unsupported provider: {provider}")
    model = (model or os.environ.get("LLM_MODEL") or "").strip() or default_model(provider)
    snippet = (text or "").strip()[:max_chars]
    if not snippet:
        raise ValueError("empty text")

    if provider == "echo":
        lid = _label_echo(snippet, seed)
    else:
        api_key = _resolve_api_key(provider)
        if not api_key:
            raise EnvironmentError(f"no API key configured for provider={provider}")
        if provider == "openai":
            lid = _label_openai(snippet, model, temperature, timeout_s, api_key)
        else:
            lid = _label_google(snippet, model, temperature, timeout_s, api_key)

    return PseudoLabel(
        label_id=int(lid),
        label_name=LABEL_ID_TO_STR[int(lid)],
        provider=provider,
        model=model,
    )
