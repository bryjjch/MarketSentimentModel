"""Model-promote Lambda: points the serving endpoint at a newly approved model.

Triggered by EventBridge when a model package in the FinSense package group moves to
``Approved`` — the manual gate at the end of the SageMaker training pipeline, which
registers every version that clears the macro-F1 threshold as ``PendingManualApproval``.

This Lambda, not Terraform, owns the SageMaker model / endpoint config / endpoint.
The model artifact changes out of band every time a pipeline run is approved, so
describing it in Terraform would mean every retrain shows up as config drift and every
deploy would need the 405 MB tarball on disk. Terraform still owns the endpoint's
*shape* — the serverless sizing arrives here as environment variables.

Also invokable directly for rollback:

    aws lambda invoke --function-name finsense-model-promote \\
        --payload '{"model_package_arn": "arn:aws:sagemaker:...:model-package/finsense-sentiment/3"}' \\
        /dev/stdout

An empty payload ``{}`` promotes the most recently approved version.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROJECT_NAME = os.environ["PROJECT_NAME"]
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
MODEL_PACKAGE_GROUP = os.environ["MODEL_PACKAGE_GROUP"]
EXECUTION_ROLE_ARN = os.environ["EXECUTION_ROLE_ARN"]
SERVERLESS_MEMORY_MB = int(os.environ["SERVERLESS_MEMORY_MB"])
SERVERLESS_MAX_CONCURRENCY = int(os.environ["SERVERLESS_MAX_CONCURRENCY"])
# How many model/endpoint-config generations to leave behind for rollback.
KEEP_VERSIONS = int(os.environ.get("KEEP_VERSIONS", "3"))

VARIANT_NAME = "primary"

_sm = boto3.client("sagemaker")

# Matches the names minted below, capturing the model package version.
_VERSION_SUFFIX = re.compile(r"-v(\d+)$")


def _already_exists(err: ClientError) -> bool:
    """True when a create_* call failed only because the resource is already there.

    EventBridge delivers at least once and retries failed invocations, so both creates
    must be no-ops on redelivery.
    """
    if err.response.get("Error", {}).get("Code") != "ValidationException":
        return False
    return "already exist" in err.response.get("Error", {}).get("Message", "").lower()


def _not_found(err: ClientError) -> bool:
    """True when describe_endpoint failed because the endpoint does not exist yet."""
    if err.response.get("Error", {}).get("Code") != "ValidationException":
        return False
    message = err.response.get("Error", {}).get("Message", "").lower()
    return "could not find" in message or "not found" in message


def _latest_approved_package() -> str:
    """ARN of the most recently approved package in the group."""
    resp = _sm.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    summaries = resp.get("ModelPackageSummaryList", [])
    if not summaries:
        raise RuntimeError(f"No approved model package in group {MODEL_PACKAGE_GROUP}")
    return summaries[0]["ModelPackageArn"]


def _resolve_package_arn(event: dict[str, Any]) -> str:
    """Pick the package to promote from an EventBridge event or a manual payload."""
    detail = event.get("detail") or {}

    arn = event.get("model_package_arn") or detail.get("ModelPackageArn")
    if arn:
        return arn

    # An event for a different group means the rule pattern is wrong; fail loudly
    # rather than silently promoting whatever happens to be newest.
    group = detail.get("ModelPackageGroupName")
    if group and group != MODEL_PACKAGE_GROUP:
        raise RuntimeError(f"Event is for group {group}, expected {MODEL_PACKAGE_GROUP}")

    return _latest_approved_package()


def _ensure_model(package_arn: str, version: int) -> str:
    name = f"{PROJECT_NAME}-model-v{version}"
    try:
        _sm.create_model(
            ModelName=name,
            # Referencing the package directly lets SageMaker resolve the container
            # image and artifact URI recorded at registration time.
            PrimaryContainer={"ModelPackageName": package_arn},
            ExecutionRoleArn=EXECUTION_ROLE_ARN,
        )
        logger.info("created_model name=%s package=%s", name, package_arn)
    except ClientError as e:
        if not _already_exists(e):
            raise
        logger.info("model_exists name=%s", name)
    return name


def _ensure_endpoint_config(model_name: str, version: int) -> str:
    name = f"{PROJECT_NAME}-ep-cfg-v{version}"
    try:
        _sm.create_endpoint_config(
            EndpointConfigName=name,
            ProductionVariants=[
                {
                    "VariantName": VARIANT_NAME,
                    "ModelName": model_name,
                    "InitialVariantWeight": 1,
                    "ServerlessConfig": {
                        "MemorySizeInMB": SERVERLESS_MEMORY_MB,
                        "MaxConcurrency": SERVERLESS_MAX_CONCURRENCY,
                    },
                }
            ],
        )
        logger.info("created_endpoint_config name=%s model=%s", name, model_name)
    except ClientError as e:
        if not _already_exists(e):
            raise
        logger.info("endpoint_config_exists name=%s", name)
    return name


def _apply_to_endpoint(config_name: str) -> str:
    """Create the endpoint on first promotion, otherwise update it in place."""
    try:
        current = _sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    except ClientError as e:
        if not _not_found(e):
            raise
        _sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        logger.info("created_endpoint name=%s config=%s", ENDPOINT_NAME, config_name)
        return "created"

    if current.get("EndpointConfigName") == config_name:
        logger.info("endpoint_already_current name=%s config=%s", ENDPOINT_NAME, config_name)
        return "unchanged"

    status = current.get("EndpointStatus")
    if status not in ("InService", "Failed"):
        # Raising lets the EventBridge async retry pick it up once the in-flight
        # update settles, instead of losing the promotion.
        raise RuntimeError(f"Endpoint {ENDPOINT_NAME} is {status}; cannot update yet")

    _sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
    logger.info(
        "updated_endpoint name=%s from=%s to=%s",
        ENDPOINT_NAME,
        current.get("EndpointConfigName"),
        config_name,
    )
    return "updated"


def _versioned_names(list_call, key: str, name_key: str, prefix: str) -> list[tuple[int, str]]:
    """Collect (version, name) for resources this Lambda minted, newest version first."""
    found: list[tuple[int, str]] = []
    paginator = _sm.get_paginator(list_call)
    for page in paginator.paginate(NameContains=prefix):
        for item in page.get(key, []):
            name = item[name_key]
            match = _VERSION_SUFFIX.search(name)
            if match and name.startswith(prefix):
                found.append((int(match.group(1)), name))
    return sorted(found, reverse=True)


def _prune(keep_config: str, keep_model: str) -> None:
    """Delete old generations, always sparing whatever the endpoint currently serves."""
    configs = _versioned_names(
        "list_endpoint_configs", "EndpointConfigs", "EndpointConfigName", f"{PROJECT_NAME}-ep-cfg-v"
    )
    for _, name in configs[KEEP_VERSIONS:]:
        if name == keep_config:
            continue
        try:
            _sm.delete_endpoint_config(EndpointConfigName=name)
            logger.info("pruned_endpoint_config name=%s", name)
        except ClientError as e:
            logger.warning("prune_endpoint_config_failed name=%s: %s", name, e)

    models = _versioned_names("list_models", "Models", "ModelName", f"{PROJECT_NAME}-model-v")
    for _, name in models[KEEP_VERSIONS:]:
        if name == keep_model:
            continue
        try:
            _sm.delete_model(ModelName=name)
            logger.info("pruned_model name=%s", name)
        except ClientError as e:
            logger.warning("prune_model_failed name=%s: %s", name, e)


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    package_arn = _resolve_package_arn(event)
    package = _sm.describe_model_package(ModelPackageName=package_arn)

    status = package.get("ModelApprovalStatus")
    if status != "Approved":
        # Rejections and re-registrations also raise state-change events.
        logger.info("skipping_unapproved package=%s status=%s", package_arn, status)
        return {"promoted": False, "reason": f"status={status}", "model_package_arn": package_arn}

    version = package["ModelPackageVersion"]

    model_name = _ensure_model(package_arn, version)
    config_name = _ensure_endpoint_config(model_name, version)
    action = _apply_to_endpoint(config_name)
    _prune(keep_config=config_name, keep_model=model_name)

    logger.info(
        "promote_complete version=%s action=%s endpoint=%s", version, action, ENDPOINT_NAME
    )
    return {
        "promoted": action != "unchanged",
        "action": action,
        "model_package_arn": package_arn,
        "model_package_version": version,
        "endpoint_name": ENDPOINT_NAME,
        "endpoint_config_name": config_name,
    }
