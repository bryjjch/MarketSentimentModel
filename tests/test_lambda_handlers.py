"""End-to-end-ish tests for the pipeline/API Lambdas with mocked AWS/SageMaker/SQS."""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

import finsense_shared.pipeline as pipeline
from finsense_shared.http import parse_json_body, parse_limit

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


class FakeSQSClient:
    """Records send_message calls; monkeypatched over finsense_shared.aws.sqs._client."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send_message(self, *, QueueUrl: str, MessageBody: str) -> dict[str, Any]:
        self.sent.append({"queue": QueueUrl, "body": json.loads(MessageBody)})
        return {"MessageId": f"m-{len(self.sent)}"}

    def sent_to(self, queue_url: str) -> list[dict[str, Any]]:
        return [s["body"] for s in self.sent if s["queue"] == queue_url]


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


class FakeConditionalTable:
    """DynamoDB table fake honouring the cache_write conditional put."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def put_item(
        self,
        *,
        Item: dict[str, Any],
        ConditionExpression: str | None = None,
        ExpressionAttributeValues: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.rows.get(Item["symbol"])
        if ConditionExpression and existing is not None:
            updated_at = (ExpressionAttributeValues or {}).get(":u", 0)
            if existing["updated_at"] > updated_at:
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
        self.rows[Item["symbol"]] = Item
        return {}


class FakeCacheReadTable:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.next_key: dict[str, Any] | None = None
        self.scan_kwargs: list[dict[str, Any]] = []

    def scan(self, **kwargs: Any) -> dict[str, Any]:
        self.scan_kwargs.append(kwargs)
        resp: dict[str, Any] = {"Items": list(self.rows.values())}
        if self.next_key:
            resp["LastEvaluatedKey"] = self.next_key
        return resp

    def get_item(self, *, Key: dict[str, str]) -> dict[str, Any]:
        item = self.rows.get(Key["symbol"])
        return {"Item": item} if item else {}


class FakeDDBResource:
    def __init__(self, table_obj: Any) -> None:
        self.table_obj = table_obj

    def Table(self, name: str) -> Any:
        return self.table_obj


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> FakeS3:
    fake = FakeS3()
    import finsense_shared.aws.s3 as s3io

    monkeypatch.setattr(s3io, "_client", lambda: fake)
    return fake


@pytest.fixture
def fake_sqs(monkeypatch: pytest.MonkeyPatch) -> FakeSQSClient:
    fake = FakeSQSClient()
    import finsense_shared.aws.sqs as sqs

    monkeypatch.setattr(sqs, "_client", lambda: fake)
    return fake


def _reload_handler(env: dict[str, str], handler_dir: str) -> Any:
    """Reload a lambda handler module with a specific env and sys.path."""
    os.environ.update(env)
    for key in ("handler",):
        sys.modules.pop(key, None)
    path = str(ROOT / "src" / "lambdas" / handler_dir)
    shared = str(ROOT / "src" / "lambdas" / "shared")
    for p in (path, shared):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module("handler")


def _sqs_event(*bodies: dict[str, Any]) -> dict[str, Any]:
    return {
        "Records": [
            {"messageId": f"msg-{i}", "body": json.dumps(b)}
            for i, b in enumerate(bodies)
        ]
    }


def test_all_lambda_sources_compile() -> None:
    """Every Lambda source must be parseable by CPython as it is stored on disk.

    The container images COPY these files verbatim, so a non-UTF-8 file builds fine and
    only surfaces in production as Runtime.UserCodeSyntaxError on import.
    """
    sources = [
        p for p in sorted((ROOT / "src" / "lambdas").rglob("*.py"))
        if "__pycache__" not in p.parts
    ]
    assert sources, "no Lambda sources found"
    for path in sources:
        try:
            compile(path.read_bytes(), str(path), "exec")
        except (SyntaxError, UnicodeDecodeError) as exc:
            pytest.fail(f"{path.relative_to(ROOT)} does not compile: {exc}")


# ---------------------------------------------------------------------------
# pipeline_dispatch
# ---------------------------------------------------------------------------


def test_dispatch_enqueues_one_collect_task_per_ticker(
    fake_sqs: FakeSQSClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "COLLECT_QUEUE_URL": "collect-q",
        },
        "pipeline_dispatch",
    )
    monkeypatch.setattr(handler, "load_tickers", lambda: ("AAPL", "MSFT", "GOOGL"))

    result = handler.lambda_handler({}, None)

    assert result["symbols"] == 3
    assert result["enqueued"] == 3
    tasks = fake_sqs.sent_to("collect-q")
    assert [t["symbol"] for t in tasks] == ["AAPL", "MSFT", "GOOGL"]
    assert all(t["task"] == "collect" for t in tasks)
    # All tasks of one dispatch share the run_id so partitions line up downstream.
    assert len({t["run_id"] for t in tasks}) == 1
    assert tasks[0]["run_id"] == result["run_id"]


def test_dispatch_single_symbol_direct_invoke(fake_sqs: FakeSQSClient) -> None:
    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "COLLECT_QUEUE_URL": "collect-q",
        },
        "pipeline_dispatch",
    )

    result = handler.lambda_handler({"symbol": "aapl", "options": {"max_articles": 5}}, None)

    assert result["symbols"] == 1
    tasks = fake_sqs.sent_to("collect-q")
    assert len(tasks) == 1
    assert tasks[0]["symbol"] == "AAPL"
    assert tasks[0]["options"]["max_articles"] == 5


# ---------------------------------------------------------------------------
# pipeline_collect
# ---------------------------------------------------------------------------


def test_collect_writes_raw_partition_and_enqueues_predict_task(
    fake_s3: FakeS3,
    fake_sqs: FakeSQSClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finsense_shared.sources.base import CollectedItem

    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "DATA_BUCKET": "data-bucket",
            "PREDICT_QUEUE_URL": "predict-q",
        },
        "pipeline_collect",
    )
    monkeypatch.setattr(
        handler,
        "collect_for_symbol",
        lambda symbol, max_articles, include_social: [
            CollectedItem(title="Beat", url="http://a", text="EPS beat expectations", source_type="news_rss"),
            CollectedItem(title="Empty", url="http://b", text="   ", source_type="news_rss"),
        ],
    )

    task = pipeline.build_collect_task("run-1", "AAPL", max_articles=5, include_social=False)
    result = handler.lambda_handler(_sqs_event(task), None)["results"][0]

    assert result["count"] == 1
    assert result["dropped_empty"] == 1
    assert result["dispatched"] is True
    assert ("data-bucket", result["key"]) in fake_s3.objects
    assert result["key"].startswith("raw/dt=")

    predict_tasks = fake_sqs.sent_to("predict-q")
    assert len(predict_tasks) == 1
    assert predict_tasks[0]["task"] == "predict"
    assert predict_tasks[0]["symbol"] == "AAPL"
    assert predict_tasks[0]["run_id"] == "run-1"
    assert predict_tasks[0]["key"] == result["key"]
    assert predict_tasks[0]["count"] == 1


def test_collect_empty_result_does_not_enqueue_predict(
    fake_s3: FakeS3,
    fake_sqs: FakeSQSClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "DATA_BUCKET": "data-bucket",
            "PREDICT_QUEUE_URL": "predict-q",
        },
        "pipeline_collect",
    )
    monkeypatch.setattr(handler, "collect_for_symbol", lambda symbol, max_articles, include_social: [])

    task = pipeline.build_collect_task("run-1", "AAPL", max_articles=5, include_social=False)
    result = handler.lambda_handler(_sqs_event(task), None)["results"][0]

    assert result["count"] == 0
    assert result["dispatched"] is False
    assert fake_sqs.sent == []
    assert fake_s3.objects == {}


# ---------------------------------------------------------------------------
# pipeline_predict
# ---------------------------------------------------------------------------


def test_predict_lambda_splits_high_and_low_confidence(
    fake_s3: FakeS3,
    fake_sqs: FakeSQSClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import finsense_shared.aws.sagemaker as sm

    fake_sm = FakeSageMakerRuntime()
    monkeypatch.setattr(sm, "_client", lambda region=None: fake_sm)

    # Seed a raw partition for the predict Lambda to read.
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
            "LABEL_QUEUE_URL": "label-q",
            "CACHE_WRITE_QUEUE_URL": "cache-write-q",
            "CACHE_TTL_SECONDS": "604800",
            "LOW_CONF_TOP_PROB": "0.65",
        },
        "pipeline_predict",
    )

    task = pipeline.build_predict_task("run-1", "AAPL", bucket="data-bucket", key=raw_key, count=4)
    result = handler.lambda_handler(_sqs_event(task), None)["results"][0]

    assert result["predictions"] == 4
    assert result["high_confidence"] == 2
    assert result["low_confidence"] == 2
    assert result["label_dispatched"] is True

    # Predictions + curated partitions were written.
    assert ("data-bucket", "predictions/dt=2026-04-23/symbol=AAPL/run-1.jsonl") in fake_s3.objects
    assert ("data-bucket", "curated/dt=2026-04-23/symbol=AAPL/run-1.jsonl") in fake_s3.objects

    # Pointer-form label task references the low-confidence rows by index only.
    label_tasks = fake_sqs.sent_to("label-q")
    assert len(label_tasks) == 1
    assert label_tasks[0]["task"] == "label"
    assert label_tasks[0]["row_indices"] == [1, 3]
    assert label_tasks[0]["predictions_key"] == result["predictions_key"]
    assert label_tasks[0]["dt"] == "2026-04-23"
    assert "rows" not in label_tasks[0]

    # The DynamoDB write is delegated to the cache_write Lambda via its queue.
    cache_tasks = fake_sqs.sent_to("cache-write-q")
    assert len(cache_tasks) == 1
    assert cache_tasks[0]["task"] == "cache_write"
    assert cache_tasks[0]["symbol"] == "AAPL"
    assert cache_tasks[0]["source"] == "pipeline"
    assert cache_tasks[0]["ttl_seconds"] == 604800
    assert cache_tasks[0]["article_count"] == result["analyzed"]


# ---------------------------------------------------------------------------
# pipeline_label
# ---------------------------------------------------------------------------


def test_label_lambda_pointer_form_reads_rows_from_predictions(fake_s3: FakeS3) -> None:
    pred_rows = [
        {"row_index": 0, "text": "Great quarter", "label_id": 2, "label_name": "positive",
         "probabilities": {"negative": 0.05, "neutral": 0.05, "positive": 0.9}, "confidence": {}},
        {"row_index": 1, "text": "Outlook unclear", "title": "Mix", "url": "http://b", "source_type": "news_rss",
         "label_id": 1, "label_name": "neutral",
         "probabilities": {"negative": 0.4, "neutral": 0.4, "positive": 0.2},
         "confidence": {"top_prob": 0.4, "margin": 0.0, "entropy": 1.0}},
    ]
    predictions_key = "predictions/dt=2026-04-23/symbol=AAPL/run-1.jsonl"
    fake_s3.objects[("data-bucket", predictions_key)] = {
        "Body": b"\n".join(json.dumps(r).encode("utf-8") for r in pred_rows)
    }

    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "DATA_BUCKET": "data-bucket",
            "LLM_PROVIDER": "echo",
            "LLM_MODEL": "",
        },
        "pipeline_label",
    )

    task = pipeline.build_label_task(
        "run-1",
        "AAPL",
        bucket="data-bucket",
        dt="2026-04-23",
        predictions_key=predictions_key,
        row_indices=[1],
    )
    result = handler.lambda_handler(_sqs_event(task), None)["results"][0]

    assert result["labeled"] == 1
    assert result["failed"] == 0
    assert result["pseudo_key"].startswith("pseudo/dt=2026-04-23/")
    assert result["curated_key"].startswith("curated/dt=2026-04-23/")
    pseudo_body = fake_s3.objects[("data-bucket", result["pseudo_key"])]["Body"]
    labeled = [json.loads(line) for line in pseudo_body.decode("utf-8").splitlines() if line.strip()]
    assert len(labeled) == 1
    assert labeled[0]["row_index"] == 1
    assert labeled[0]["model_label_name"] == "neutral"


def test_label_lambda_inline_rows_replay_writes_pseudo_and_curated(fake_s3: FakeS3) -> None:
    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "DATA_BUCKET": "data-bucket",
            "LLM_PROVIDER": "echo",
            "LLM_MODEL": "",
        },
        "pipeline_label",
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


# ---------------------------------------------------------------------------
# cache_write
# ---------------------------------------------------------------------------


def _cache_write_handler(monkeypatch: pytest.MonkeyPatch, table: FakeConditionalTable) -> Any:
    import boto3

    monkeypatch.setattr(boto3, "resource", lambda name, **kwargs: FakeDDBResource(table))
    return _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "TABLE_NAME": "sentiment-cache",
        },
        "cache_write",
    )


def test_cache_write_puts_item_with_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    table = FakeConditionalTable()
    handler = _cache_write_handler(monkeypatch, table)

    task = pipeline.build_cache_write_task(
        "AAPL",
        score=0.42,
        label="positive",
        article_count=9,
        recent_headlines=[{"title": "Beat", "url": "http://a"}],
        updated_at=1_753_459_200,
        ttl_seconds=600,
        source="api",
    )
    result = handler.lambda_handler(_sqs_event(task), None)

    assert result == {"batchItemFailures": []}
    row = table.rows["AAPL"]
    assert row["label"] == "positive"
    assert float(row["score"]) == 0.42
    assert row["article_count"] == 9
    assert row["updated_at"] == 1_753_459_200
    assert row["expires_at"] == 1_753_459_200 + 600


def test_cache_write_skips_stale_message_without_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    table = FakeConditionalTable()
    table.rows["AAPL"] = {"symbol": "AAPL", "label": "positive", "updated_at": 2_000_000_000}
    handler = _cache_write_handler(monkeypatch, table)

    stale = pipeline.build_cache_write_task(
        "AAPL",
        score=-0.5,
        label="negative",
        article_count=1,
        recent_headlines=[],
        updated_at=1_000_000_000,
        ttl_seconds=600,
        source="pipeline",
    )
    result = handler.lambda_handler(_sqs_event(stale), None)

    # Stale writes are dropped silently — no retry, no overwrite.
    assert result == {"batchItemFailures": []}
    assert table.rows["AAPL"]["label"] == "positive"


def test_cache_write_reports_partial_batch_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    table = FakeConditionalTable()
    handler = _cache_write_handler(monkeypatch, table)

    good = pipeline.build_cache_write_task(
        "AAPL",
        score=0.1,
        label="neutral",
        article_count=2,
        recent_headlines=[],
        updated_at=1_753_459_200,
        ttl_seconds=600,
        source="pipeline",
    )
    bad = {"task": "cache_write", "symbol": "MSFT"}  # missing required fields
    result = handler.lambda_handler(_sqs_event(good, bad), None)

    assert result == {"batchItemFailures": [{"itemIdentifier": "msg-1"}]}
    assert "AAPL" in table.rows
    assert "MSFT" not in table.rows


# ---------------------------------------------------------------------------
# api_sentiment
# ---------------------------------------------------------------------------


def _api_sentiment_handler(monkeypatch: pytest.MonkeyPatch, runtime: Any) -> Any:
    import finsense_shared.aws.sagemaker as sm

    monkeypatch.setattr(sm, "_client", lambda region=None: runtime)
    return _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "SAGEMAKER_ENDPOINT_NAME": "test-endpoint",
            "CACHE_WRITE_QUEUE_URL": "cache-write-q",
            "CACHE_TTL_SECONDS": "600",
        },
        "api_sentiment",
    )


def test_api_sentiment_success_enqueues_cache_write(
    fake_sqs: FakeSQSClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finsense_shared.sources.base import CollectedItem

    handler = _api_sentiment_handler(monkeypatch, FakeSageMakerRuntimeForApi())
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

    cache_tasks = fake_sqs.sent_to("cache-write-q")
    assert len(cache_tasks) == 1, "expected a cache-write task"
    task = cache_tasks[0]
    assert task["task"] == "cache_write"
    assert task["symbol"] == "AAPL"
    assert task["label"] == out["label"]
    assert task["article_count"] == out["article_count"]
    assert task["updated_at"] == out["updated_at"]
    assert task["ttl_seconds"] == 600
    assert task["source"] == "api"


def test_api_sentiment_error_does_not_enqueue_cache_write(
    fake_sqs: FakeSQSClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finsense_shared.sources.base import CollectedItem

    class InvalidShapeRuntime:
        def invoke_endpoint(
            self, *, EndpointName: str, ContentType: str, Accept: str, Body: bytes
        ) -> dict[str, Any]:
            return {"Body": io.BytesIO(json.dumps({"unexpected": "shape"}).encode("utf-8"))}

    handler = _api_sentiment_handler(monkeypatch, InvalidShapeRuntime())
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
    assert out["error"] == "sagemaker_batch_size_mismatch"
    assert fake_sqs.sent == []


def test_api_sentiment_rejects_unknown_ticker(
    fake_sqs: FakeSQSClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _api_sentiment_handler(monkeypatch, FakeSageMakerRuntimeForApi())

    resp = handler.lambda_handler(
        {
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps({"symbol": "ZZZZZ"}),
            "isBase64Encoded": False,
        },
        None,
    )
    assert resp["statusCode"] == 400
    body = json.loads(resp["body"])
    assert body["error"] == "invalid_symbol"
    assert "Unknown ticker symbol" in body["message"]
    assert fake_sqs.sent == []


# ---------------------------------------------------------------------------
# api_cache_read
# ---------------------------------------------------------------------------


def _cache_read_handler(monkeypatch: pytest.MonkeyPatch, table: FakeCacheReadTable) -> Any:
    import boto3

    monkeypatch.setattr(boto3, "resource", lambda name, **kwargs: FakeDDBResource(table))
    return _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "TABLE_NAME": "sentiment-cache",
        },
        "api_cache_read",
    )


def test_cache_read_list_clamps_limit_and_roundtrips_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    table = FakeCacheReadTable()
    table.rows["AAPL"] = {"symbol": "AAPL", "label": "positive", "updated_at": 1, "expires_at": 9_999_999_999}
    table.next_key = {"symbol": "AAPL"}
    handler = _cache_read_handler(monkeypatch, table)

    resp = handler.lambda_handler({"queryStringParameters": {"limit": "9999"}, "pathParameters": {}}, None)
    assert resp["statusCode"] == 200
    assert table.scan_kwargs[0]["Limit"] == 500
    assert json.loads(resp["body"])[0]["symbol"] == "AAPL"

    cursor = resp["headers"]["X-Next-Cursor"]
    table.next_key = None
    resp2 = handler.lambda_handler({"queryStringParameters": {"cursor": cursor}, "pathParameters": {}}, None)
    assert resp2["statusCode"] == 200
    assert table.scan_kwargs[1]["ExclusiveStartKey"] == {"symbol": "AAPL"}
    assert "X-Next-Cursor" not in resp2["headers"]


def test_cache_read_list_rejects_bad_input(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _cache_read_handler(monkeypatch, FakeCacheReadTable())

    resp = handler.lambda_handler({"queryStringParameters": {"cursor": "!!!"}, "pathParameters": {}}, None)
    assert resp["statusCode"] == 400
    assert json.loads(resp["body"])["message"] == "cursor is invalid"

    resp = handler.lambda_handler({"queryStringParameters": {"limit": "0"}, "pathParameters": {}}, None)
    assert resp["statusCode"] == 400
    assert json.loads(resp["body"])["message"] == "limit must be >= 1"


def test_cache_read_get_by_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    table = FakeCacheReadTable()
    table.rows["AAPL"] = {"symbol": "AAPL", "label": "positive", "updated_at": 1, "expires_at": 9_999_999_999}
    table.rows["MSFT"] = {"symbol": "MSFT", "label": "neutral", "updated_at": 1, "expires_at": 2}
    handler = _cache_read_handler(monkeypatch, table)

    resp = handler.lambda_handler({"pathParameters": {"symbol": "aapl"}}, None)
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["symbol"] == "AAPL"

    # TTL-expired rows 404 even before DynamoDB physically deletes them.
    resp = handler.lambda_handler({"pathParameters": {"symbol": "MSFT"}}, None)
    assert resp["statusCode"] == 404

    resp = handler.lambda_handler({"pathParameters": {"symbol": "NVDA"}}, None)
    assert resp["statusCode"] == 404


# ---------------------------------------------------------------------------
# api_ticker_suggest
# ---------------------------------------------------------------------------


def test_ticker_suggestions_prefix() -> None:
    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "VALID_TICKERS_JSON": "[\"AAPL\", \"AAP\", \"MSFT\"]",
            "VALID_TICKERS_SSM_PARAM": "",
        },
        "api_ticker_suggest",
    )
    resp = handler.lambda_handler(
        {"queryStringParameters": {"q": "aa", "limit": "5"}},
        None,
    )
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["query"] == "AA"
    assert body["suggestions"] == ["AAP", "AAPL"]


def test_ticker_suggestions_requires_query() -> None:
    handler = _reload_handler(
        {
            "AWS_DEFAULT_REGION": "us-east-1",
            "VALID_TICKERS_JSON": "[\"AAPL\"]",
        },
        "api_ticker_suggest",
    )
    resp = handler.lambda_handler({"queryStringParameters": {}}, None)
    assert resp["statusCode"] == 400


# ---------------------------------------------------------------------------
# finsense_shared.http / finsense_shared.pipeline units
# ---------------------------------------------------------------------------


def test_parse_json_body_handles_base64() -> None:
    import base64

    raw = json.dumps({"symbol": "AAPL"})
    event = {"body": base64.b64encode(raw.encode("utf-8")).decode("ascii"), "isBase64Encoded": True}
    assert parse_json_body(event) == {"symbol": "AAPL"}
    assert parse_json_body({"body": "not json"}) == {}
    assert parse_json_body({"body": None}) == {}


def test_parse_limit_validation() -> None:
    assert parse_limit({}, default=10, maximum=25) == 10
    assert parse_limit({"limit": "100"}, default=10, maximum=25) == 25
    with pytest.raises(ValueError):
        parse_limit({"limit": "abc"}, default=10, maximum=25)
    with pytest.raises(ValueError):
        parse_limit({"limit": "-1"}, default=10, maximum=25)


def test_validate_task_rejects_malformed_payloads() -> None:
    good = pipeline.build_predict_task("run-1", "AAPL", bucket="b", key="k", count=1)
    assert pipeline.validate_task(good, pipeline.TASK_PREDICT) is good
    with pytest.raises(ValueError):
        pipeline.validate_task(good, pipeline.TASK_COLLECT)
    with pytest.raises(ValueError):
        pipeline.validate_task({"task": "predict", "run_id": "run-1"}, pipeline.TASK_PREDICT)
    with pytest.raises(ValueError):
        pipeline.validate_task("not a dict", pipeline.TASK_PREDICT)
