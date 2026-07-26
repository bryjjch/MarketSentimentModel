"""Thin S3 helpers for reading/writing JSON Lines partitions."""

from __future__ import annotations

import io
import json
from typing import Any, Iterable, Iterator

import boto3


def _client() -> Any:
    return boto3.client("s3")


def write_jsonl(
    bucket: str,
    key: str,
    records: Iterable[dict[str, Any]],
    *,
    client: Any | None = None,
    extra_args: dict[str, Any] | None = None,
) -> int:
    """Write ``records`` as JSON Lines to ``s3://bucket/key``. Returns the row count written."""
    c = client or _client()
    buf = io.BytesIO()
    count = 0
    for rec in records:
        buf.write(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
        buf.write(b"\n")
        count += 1
    body = buf.getvalue()
    put_kwargs: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": body,
        "ContentType": "application/x-ndjson",
        "ServerSideEncryption": "AES256",
    }
    if extra_args:
        put_kwargs.update(extra_args)
    c.put_object(**put_kwargs)
    return count


def read_jsonl(
    bucket: str,
    key: str,
    *,
    client: Any | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per line from ``s3://bucket/key`` (skip blank/unparseable lines)."""
    c = client or _client()
    resp = c.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read().decode("utf-8", errors="replace")
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj
