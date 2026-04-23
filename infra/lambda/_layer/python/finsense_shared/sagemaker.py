"""Batched SageMaker InvokeEndpoint helper shared across Lambdas."""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _client(region: str | None = None) -> Any:
    if region:
        return boto3.client("sagemaker-runtime", region_name=region)
    return boto3.client("sagemaker-runtime")


def invoke_predict(
    endpoint_name: str,
    texts: list[str],
    *,
    batch_size: int = 32,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Call ``InvokeEndpoint`` with ``{"texts": [...]}`` in batches and concatenate records in input order.

    The SageMaker handler (``infra/sagemaker/serving/code/inference.py``) returns a list of
    per-text records of shape::

        {"text": "...", "label_id": 0|1|2, "label_name": "negative|neutral|positive",
         "probabilities": {"negative": .., "neutral": .., "positive": ..}, "error": null}

    Rows that fail parsing or are missing/short are returned with ``error`` populated so callers
    can decide whether to route them for pseudo-labeling.
    """
    if not texts:
        return []

    c = client or _client()
    batch_size = max(1, min(batch_size, 64))
    out: list[dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        payload = json.dumps({"texts": chunk}).encode("utf-8")
        try:
            resp = c.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=payload,
            )
        except ClientError as e:
            err = e.response.get("Error", {})
            code = err.get("Code", "InvocationError")
            msg = err.get("Message", str(e))
            logger.warning("sagemaker_invoke_failed code=%s msg=%s", code, msg)
            for t in chunk:
                out.append({"text": t, "error": code, "message": msg})
            continue

        raw = resp["Body"].read().decode("utf-8")
        try:
            records = json.loads(raw)
        except json.JSONDecodeError:
            for t in chunk:
                out.append({"text": t, "error": "invalid_sagemaker_response"})
            continue

        if not isinstance(records, list) or len(records) != len(chunk):
            for t in chunk:
                out.append({"text": t, "error": "sagemaker_batch_size_mismatch"})
            continue

        out.extend(records)

    return out
