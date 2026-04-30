"""End-to-end-ish tests for the ingestion/prediction/pseudo-label Lambdas with mocked AWS/SageMaker."""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = dict(kwargs)
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        body = self.objects[(Bucket, Key)]["Body"]
        return {"Body": io.BytesIO(body)}


class FakeSageMakerRuntime:
    """Canned responses keyed on input batch length; emits alternating high/low confidence."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke_endpoint(self, *, EndpointName: str, ContentType: str, Accept: str, Body: bytes) -> dict[str, Any]:
        self.calls.append({"endpoint": EndpointName, "body": Body})
        payload = json.loads(Body.decode("utf-8"))
        texts = payload["texts"]
        records = []
        for i, t in enumerate(texts):
            if i % 2 == 0:
                records.append({
                    "text": t,
                    "label_id": 2,
                    "label_name": "positive",
                    "probabilities": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
                })
            else:
                records.append({
                    "text": t,
                    "label_id": 1,
                    "label_name": "neutral",
                    "probabilities": {"negative": 0.4, "neutral": 0.4, "positive": 0.2},
                })
        return {"Body": io.BytesIO(json.dumps(records).encode("utf-8"))}


class FakeLambda:
    def __init__(self) -> None:
        self.invocations: list[dict[str, Any]] = []

    def invoke(self, *, FunctionName: str, InvocationType: str, Payload: bytes) -> dict[str, Any]:
        self.invocations.append({
            "name": FunctionName,
            "type": InvocationType,
            "payload": json.loads(Payload.decode("utf-8")),
        })
        return {"StatusCode": 202, "Payload": io.BytesIO(b"")}


class FakeSageMakerRuntimeForApi:
    """Simple API runtime stub that returns one positive record per text."""

    def invoke_endpoint(self, *, EndpointName: str, ContentType: str, Accept: str, Body: bytes) -> dict[str, Any]:
        payload = json.loads(Body.decode("utf-8"))
        texts = payload.get("texts", [])
        records = [
            {
                "label_id": 2,
                "label_name": "positive",
                "probabilities": {"negative": 0.05, "neutral": 0.1, "positive": 0.85},
            }
            for _ in texts
        ]
        return {"Body": io.BytesIO(json.dumps(records).encode("utf-8"))}


class FakeDDBTable:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put_item(self, *, Item: dict[str, Any]) -> dict[str, Any]:
        self.items.append(Item)
        return {}


class FakeDDBResource:
    def __init__(self) -> None:
        self.table_obj = FakeDDBTable()

    def Table(self, name: str) -> FakeDDBTable:
        return self.table_obj


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> FakeS3:
    fake = FakeS3()
    import finsense_shared.s3io as s3io

    monkeypatch.setattr(s3io, "_client", lambda: fake)
    return fake


def _reload_handler(env: dict[str, str], handler_dir: str) -> Any:
    """Reload a lambda handler module with a specific env and sys.path."""
    os.environ.update(env)
    for key in ("handler",):
        sys.modules.pop(key, None)
    path = str(ROOT / "infra" / "lambda" / handler_dir)
    shared = str(ROOT / "infra" / "lambda" / "_layer" / "python")
    for p in (path, shared):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module("handler")


def test_prediction_lambda_splits_high_and_low_confidence(
    fake_s3: FakeS3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boto3
    import finsense_shared.sagemaker as sm

    fake_sm = FakeSageMakerRuntime()
    monkeypatch.setattr(sm, "_client", lambda region=None: fake_sm)

    fake_lambda = FakeLambda()
    fake_ddb = FakeDDBResource()

    def boto_client(name: str, **kwargs: Any) -> Any:
        if name == "lambda":
            return fake_lambda
        raise AssertionError(f"unexpected boto3.client({name})")

    monkeypatch.setattr(boto3, "client", boto_client)
    monkeypatch.setattr(boto3, "resource", lambda name, **kwargs: fake_ddb)

    # Seed a raw partition for the ingestion-prediction Lambda to read.
    rows = [
        {"text": "EPS beat expectations", "title": "Beat", "url": "http://a", "source_type": "news_rss"},
        {"text": "Outlook unclear mixed", "title": "Mix", "url": "http://b", "source_type": "news_rss"},
        {"text": "Record revenue quarter", "title": "Rev", "url": "http://c", "source_type": "news_rss"},
        {"text": "Guidance trimmed modestly", "title": "Gd", "url": "http://d", "source_type": "news_rss"},
    ]
    raw_key = "raw/dt=2026-04-23/symbol=AAPL/run-1.jsonl"
    fake_s3.objects[("data-bucket", raw_key)] = {
        "Body": b"\n".join(json.dumps(r).encode("utf-8") for r in rows)
    }

    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "DATA_BUCKET": "data-bucket",
            "SAGEMAKER_ENDPOINT_NAME": "test-endpoint",
            "PSEUDO_LABEL_FUNCTION_NAME": "test-pseudo-label",
            "CACHE_TABLE_NAME": "test-cache",
            "LOW_CONF_TOP_PROB": "0.65",
        },
        "ingestion_prediction",
    )

    result = handler.lambda_handler(
        {
            "run_id": "run-1",
            "symbol": "AAPL",
            "bucket": "data-bucket",
            "key": raw_key,
        },
        None,
    )

    assert result["predictions"] == 4
    assert result["high_confidence"] == 2
    assert result["low_confidence"] == 2
    assert result["pseudo_dispatched"] is True

    # Predictions + curated partitions were written
    assert ("data-bucket", "predictions/dt=2026-04-23/symbol=AAPL/run-1.jsonl") in fake_s3.objects
    assert ("data-bucket", "curated/dt=2026-04-23/symbol=AAPL/run-1.jsonl") in fake_s3.objects
    # Pseudo-label Lambda was fanned out with the low-confidence subset only.
    assert len(fake_lambda.invocations) == 1
    dispatched = fake_lambda.invocations[0]
    assert dispatched["name"] == "test-pseudo-label"
    assert dispatched["type"] == "Event"
    assert len(dispatched["payload"]["rows"]) == 2
    # DynamoDB cache row was written.
    assert fake_ddb.table_obj.items, "expected a cache row"
    assert fake_ddb.table_obj.items[0]["symbol"] == "AAPL"


def test_pseudo_label_lambda_echo_provider_writes_pseudo_and_curated(
    fake_s3: FakeS3,
) -> None:
    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "DATA_BUCKET": "data-bucket",
            "LLM_PROVIDER": "echo",
            "LLM_MODEL": "",
        },
        "pseudo_label",
    )

    result = handler.lambda_handler(
        {
            "run_id": "run-1",
            "symbol": "AAPL",
            "rows": [
                {
                    "row_index": 1,
                    "text": "Outlook unclear",
                    "model_label_id": 1,
                    "model_label_name": "neutral",
                    "probabilities": {"negative": 0.4, "neutral": 0.4, "positive": 0.2},
                    "confidence": {"top_prob": 0.4, "margin": 0.0, "entropy": 1.0},
                    "title": "Mix",
                    "url": "http://b",
                    "source_type": "news_rss",
                }
            ],
        },
        None,
    )

    assert result["labeled"] == 1
    assert result["failed"] == 0
    assert result["pseudo_key"].startswith("pseudo/dt=")
    assert result["curated_key"].startswith("curated/dt=")
    assert ("data-bucket", result["pseudo_key"]) in fake_s3.objects
    assert ("data-bucket", result["curated_key"]) in fake_s3.objects
    curated_body = fake_s3.objects[("data-bucket", result["curated_key"])]["Body"]
    curated = [json.loads(line) for line in curated_body.decode("utf-8").splitlines() if line.strip()]
    assert curated[0]["source"] == "pseudo"
    assert curated[0]["pseudo_provider"] == "echo"


def test_api_sentiment_by_symbol_success_writes_cache_row(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3
    from finsense_shared.sources.base import CollectedItem

    fake_runtime = FakeSageMakerRuntimeForApi()
    fake_ddb = FakeDDBResource()

    def boto_client(name: str, **kwargs: Any) -> Any:
        if name == "sagemaker-runtime":
            return fake_runtime
        raise AssertionError(f"unexpected boto3.client({name})")

    def boto_resource(name: str, **kwargs: Any) -> Any:
        if name == "dynamodb":
            return fake_ddb
        raise AssertionError(f"unexpected boto3.resource({name})")

    monkeypatch.setattr(boto3, "client", boto_client)
    monkeypatch.setattr(boto3, "resource", boto_resource)

    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "SAGEMAKER_ENDPOINT_NAME": "test-endpoint",
            "CACHE_TABLE_NAME": "test-cache",
            "CACHE_TTL_SECONDS": "600",
        },
        "api_sentiment_by_symbol",
    )
    monkeypatch.setattr(
        handler,
        "collect_for_symbol",
        lambda symbol, max_articles, include_social: [
            CollectedItem(
                title="Strong quarter",
                url="http://example.com/news",
                text="Company raised guidance and beat revenue.",
                source_type="news_rss",
            )
        ],
    )

    out = handler.run_sentiment("AAPL", {"max_articles": 5, "include_social": False})
    assert out.get("error") is None
    assert out["symbol"] == "AAPL"
    assert fake_ddb.table_obj.items, "expected a cache row"
    row = fake_ddb.table_obj.items[0]
    assert row["symbol"] == "AAPL"
    assert row["label"] == out["label"]
    assert int(row["article_count"]) == out["article_count"]
    assert int(row["expires_at"]) > int(row["updated_at"])


def test_api_sentiment_by_symbol_error_does_not_write_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3
    from finsense_shared.sources.base import CollectedItem

    fake_ddb = FakeDDBResource()

    class InvalidShapeRuntime:
        def invoke_endpoint(
            self, *, EndpointName: str, ContentType: str, Accept: str, Body: bytes
        ) -> dict[str, Any]:
            return {"Body": io.BytesIO(json.dumps({"unexpected": "shape"}).encode("utf-8"))}

    def boto_client(name: str, **kwargs: Any) -> Any:
        if name == "sagemaker-runtime":
            return InvalidShapeRuntime()
        raise AssertionError(f"unexpected boto3.client({name})")

    def boto_resource(name: str, **kwargs: Any) -> Any:
        if name == "dynamodb":
            return fake_ddb
        raise AssertionError(f"unexpected boto3.resource({name})")

    monkeypatch.setattr(boto3, "client", boto_client)
    monkeypatch.setattr(boto3, "resource", boto_resource)

    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "SAGEMAKER_ENDPOINT_NAME": "test-endpoint",
            "CACHE_TABLE_NAME": "test-cache",
            "CACHE_TTL_SECONDS": "600",
        },
        "api_sentiment_by_symbol",
    )
    monkeypatch.setattr(
        handler,
        "collect_for_symbol",
        lambda symbol, max_articles, include_social: [
            CollectedItem(
                title="Mixed outlook",
                url="http://example.com/mixed",
                text="Outlook is uncertain this quarter.",
                source_type="news_rss",
            )
        ],
    )

    out = handler.run_sentiment("AAPL", {"max_articles": 5, "include_social": False})
    assert out["error"] == "invalid_sagemaker_shape"
    assert fake_ddb.table_obj.items == []
