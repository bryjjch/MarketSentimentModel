"""Thin SQS helpers for the pipeline Lambdas (send, consume, partial-batch failures)."""

from __future__ import annotations

import json
from typing import Any, Iterator

import boto3


def _client() -> Any:
    return boto3.client("sqs")


def send_json(queue_url: str, payload: dict[str, Any], *, client: Any | None = None) -> None:
    c = client or _client()
    c.send_message(QueueUrl=queue_url, MessageBody=json.dumps(payload))


def iter_records(event: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any] | None]]:
    """Yield ``(message_id, parsed_body)`` per SQS record; body is ``None`` when unparseable."""
    for rec in event.get("Records") or []:
        if not isinstance(rec, dict):
            continue
        message_id = str(rec.get("messageId") or "")
        try:
            body = json.loads(rec.get("body") or "")
        except (json.JSONDecodeError, TypeError):
            body = None
        yield message_id, body if isinstance(body, dict) else None


def batch_failures(message_ids: list[str]) -> dict[str, Any]:
    """ReportBatchItemFailures response shape for partial-batch SQS event sources."""
    return {"batchItemFailures": [{"itemIdentifier": mid} for mid in message_ids if mid]}
