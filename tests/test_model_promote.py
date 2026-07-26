"""Tests for the model_promote Lambda, which owns the endpoint outside Terraform."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]

ENV = {
    "PROJECT_NAME": "finsense",
    "ENDPOINT_NAME": "finsense-endpoint",
    "MODEL_PACKAGE_GROUP": "finsense-sentiment",
    "EXECUTION_ROLE_ARN": "arn:aws:iam::123456789012:role/finsense-sagemaker-exec",
    "SERVERLESS_MEMORY_MB": "3072",
    "SERVERLESS_MAX_CONCURRENCY": "5",
    "KEEP_VERSIONS": "3",
}

GROUP_ARN_PREFIX = "arn:aws:sagemaker:us-east-1:123456789012:model-package/finsense-sentiment"


def _validation_error(operation: str, message: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": message}}, operation
    )


class FakeSageMaker:
    """In-memory stand-in for the SageMaker control-plane client."""

    def __init__(self, packages: dict[int, str] | None = None) -> None:
        # version -> approval status
        self.packages = packages or {1: "Approved"}
        self.models: list[str] = []
        self.endpoint_configs: list[str] = []
        self.endpoint: dict[str, Any] | None = None
        self.deleted_models: list[str] = []
        self.deleted_configs: list[str] = []
        self.updates: list[str] = []

    # --- model packages ---------------------------------------------------

    def _version_of(self, arn: str) -> int:
        return int(arn.rsplit("/", 1)[1])

    def list_model_packages(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["ModelApprovalStatus"] == "Approved"
        approved = sorted(
            (v for v, s in self.packages.items() if s == "Approved"), reverse=True
        )
        return {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": f"{GROUP_ARN_PREFIX}/{v}"} for v in approved
            ]
        }

    def describe_model_package(self, *, ModelPackageName: str) -> dict[str, Any]:
        version = self._version_of(ModelPackageName)
        return {
            "ModelPackageVersion": version,
            "ModelApprovalStatus": self.packages[version],
        }

    # --- models -----------------------------------------------------------

    def create_model(self, *, ModelName: str, **kwargs: Any) -> dict[str, Any]:
        if ModelName in self.models:
            raise _validation_error("CreateModel", f"Cannot create already existing model {ModelName}")
        self.models.append(ModelName)
        return {}

    def delete_model(self, *, ModelName: str) -> dict[str, Any]:
        self.models.remove(ModelName)
        self.deleted_models.append(ModelName)
        return {}

    # --- endpoint configs -------------------------------------------------

    def create_endpoint_config(self, *, EndpointConfigName: str, **kwargs: Any) -> dict[str, Any]:
        if EndpointConfigName in self.endpoint_configs:
            raise _validation_error(
                "CreateEndpointConfig",
                f"Cannot create already existing endpoint configuration {EndpointConfigName}",
            )
        self.endpoint_configs.append(EndpointConfigName)
        return {}

    def delete_endpoint_config(self, *, EndpointConfigName: str) -> dict[str, Any]:
        self.endpoint_configs.remove(EndpointConfigName)
        self.deleted_configs.append(EndpointConfigName)
        return {}

    # --- endpoint ---------------------------------------------------------

    def describe_endpoint(self, *, EndpointName: str) -> dict[str, Any]:
        if self.endpoint is None:
            raise _validation_error(
                "DescribeEndpoint", f"Could not find endpoint {EndpointName}"
            )
        return self.endpoint

    def create_endpoint(self, *, EndpointName: str, EndpointConfigName: str) -> dict[str, Any]:
        self.endpoint = {
            "EndpointName": EndpointName,
            "EndpointConfigName": EndpointConfigName,
            "EndpointStatus": "Creating",
        }
        return {}

    def update_endpoint(self, *, EndpointName: str, EndpointConfigName: str) -> dict[str, Any]:
        self.updates.append(EndpointConfigName)
        self.endpoint = {
            "EndpointName": EndpointName,
            "EndpointConfigName": EndpointConfigName,
            "EndpointStatus": "Updating",
        }
        return {}

    # --- pagination -------------------------------------------------------

    def get_paginator(self, operation: str) -> Any:
        key, names = {
            "list_models": ("Models", lambda: [{"ModelName": n} for n in self.models]),
            "list_endpoint_configs": (
                "EndpointConfigs",
                lambda: [{"EndpointConfigName": n} for n in self.endpoint_configs],
            ),
        }[operation]

        class _Paginator:
            def paginate(self, *, NameContains: str) -> Any:
                return [{key: [i for i in names() if NameContains in next(iter(i.values()))]}]

        return _Paginator()


@pytest.fixture
def promote(monkeypatch: pytest.MonkeyPatch):
    """Import the handler fresh with a fake SageMaker client bound at module scope."""

    def _load(fake: FakeSageMaker):
        for key, value in ENV.items():
            monkeypatch.setenv(key, value)

        import boto3

        monkeypatch.setattr(boto3, "client", lambda service, **kw: fake)

        sys.modules.pop("handler", None)
        path = str(ROOT / "src" / "lambdas" / "model_promote")
        monkeypatch.syspath_prepend(path)
        module = importlib.import_module("handler")
        sys.modules.pop("handler", None)  # don't leak into other handler tests
        return module

    return _load


def _event(version: int, status: str = "Approved") -> dict[str, Any]:
    return {
        "detail": {
            "ModelPackageArn": f"{GROUP_ARN_PREFIX}/{version}",
            "ModelPackageGroupName": "finsense-sentiment",
            "ModelApprovalStatus": status,
        }
    }


def test_first_approval_creates_the_endpoint(promote):
    fake = FakeSageMaker({1: "Approved"})
    handler = promote(fake)

    result = handler.lambda_handler(_event(1), None)

    assert result["action"] == "created"
    assert fake.models == ["finsense-model-v1"]
    assert fake.endpoint_configs == ["finsense-ep-cfg-v1"]
    assert fake.endpoint["EndpointConfigName"] == "finsense-ep-cfg-v1"


def test_later_approval_updates_the_endpoint(promote):
    fake = FakeSageMaker({1: "Approved", 2: "Approved"})
    fake.models = ["finsense-model-v1"]
    fake.endpoint_configs = ["finsense-ep-cfg-v1"]
    fake.endpoint = {
        "EndpointName": "finsense-endpoint",
        "EndpointConfigName": "finsense-ep-cfg-v1",
        "EndpointStatus": "InService",
    }
    handler = promote(fake)

    result = handler.lambda_handler(_event(2), None)

    assert result["action"] == "updated"
    assert fake.updates == ["finsense-ep-cfg-v2"]


def test_unapproved_package_is_ignored(promote):
    """Rejections raise the same state-change event type."""
    fake = FakeSageMaker({1: "Rejected"})
    handler = promote(fake)

    result = handler.lambda_handler(_event(1, "Rejected"), None)

    assert result["promoted"] is False
    assert fake.models == []
    assert fake.endpoint is None


def test_redelivery_is_idempotent(promote):
    """EventBridge delivers at least once; a second delivery must not churn the endpoint."""
    fake = FakeSageMaker({1: "Approved"})
    handler = promote(fake)

    handler.lambda_handler(_event(1), None)
    result = handler.lambda_handler(_event(1), None)

    assert result["action"] == "unchanged"
    assert fake.models == ["finsense-model-v1"]
    assert fake.endpoint_configs == ["finsense-ep-cfg-v1"]
    assert fake.updates == []


def test_empty_payload_promotes_latest_approved(promote):
    fake = FakeSageMaker({1: "Approved", 2: "Rejected", 3: "Approved"})
    handler = promote(fake)

    result = handler.lambda_handler({}, None)

    assert result["model_package_version"] == 3


def test_explicit_arn_enables_rollback(promote):
    """Re-promoting an older approved version is how a bad model is backed out."""
    fake = FakeSageMaker({1: "Approved", 2: "Approved"})
    fake.models = ["finsense-model-v1", "finsense-model-v2"]
    fake.endpoint_configs = ["finsense-ep-cfg-v1", "finsense-ep-cfg-v2"]
    fake.endpoint = {
        "EndpointName": "finsense-endpoint",
        "EndpointConfigName": "finsense-ep-cfg-v2",
        "EndpointStatus": "InService",
    }
    handler = promote(fake)

    result = handler.lambda_handler({"model_package_arn": f"{GROUP_ARN_PREFIX}/1"}, None)

    assert result["action"] == "updated"
    assert fake.updates == ["finsense-ep-cfg-v1"]


def test_event_for_another_group_is_rejected(promote):
    fake = FakeSageMaker({1: "Approved"})
    handler = promote(fake)

    event = _event(1)
    del event["detail"]["ModelPackageArn"]
    event["detail"]["ModelPackageGroupName"] = "some-other-group"

    with pytest.raises(RuntimeError, match="some-other-group"):
        handler.lambda_handler(event, None)


def test_update_while_endpoint_is_busy_raises_for_retry(promote):
    fake = FakeSageMaker({1: "Approved", 2: "Approved"})
    fake.endpoint = {
        "EndpointName": "finsense-endpoint",
        "EndpointConfigName": "finsense-ep-cfg-v1",
        "EndpointStatus": "Updating",
    }
    handler = promote(fake)

    with pytest.raises(RuntimeError, match="Updating"):
        handler.lambda_handler(_event(2), None)


def test_pruning_keeps_recent_generations_and_the_live_one(promote):
    fake = FakeSageMaker({v: "Approved" for v in range(1, 7)})
    fake.models = [f"finsense-model-v{v}" for v in range(1, 6)]
    fake.endpoint_configs = [f"finsense-ep-cfg-v{v}" for v in range(1, 6)]
    fake.endpoint = {
        "EndpointName": "finsense-endpoint",
        "EndpointConfigName": "finsense-ep-cfg-v5",
        "EndpointStatus": "InService",
    }
    handler = promote(fake)

    handler.lambda_handler(_event(6), None)

    # v6 promoted, KEEP_VERSIONS=3 leaves v6/v5/v4; v3 and below go.
    assert sorted(fake.deleted_configs) == ["finsense-ep-cfg-v1", "finsense-ep-cfg-v2", "finsense-ep-cfg-v3"]
    assert sorted(fake.deleted_models) == ["finsense-model-v1", "finsense-model-v2", "finsense-model-v3"]
    assert "finsense-ep-cfg-v6" in fake.endpoint_configs


def test_pruning_never_deletes_the_config_just_deployed(promote):
    """KEEP_VERSIONS=1 is the edge case: the new generation must survive its own prune."""
    fake = FakeSageMaker({1: "Approved", 2: "Approved"})
    fake.models = ["finsense-model-v1"]
    fake.endpoint_configs = ["finsense-ep-cfg-v1"]
    fake.endpoint = {
        "EndpointName": "finsense-endpoint",
        "EndpointConfigName": "finsense-ep-cfg-v1",
        "EndpointStatus": "InService",
    }
    handler = promote(fake)
    handler.KEEP_VERSIONS = 1

    handler.lambda_handler(_event(2), None)

    assert "finsense-ep-cfg-v2" in fake.endpoint_configs
    assert "finsense-model-v2" in fake.models
