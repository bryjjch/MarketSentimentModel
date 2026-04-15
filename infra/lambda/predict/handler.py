"""API Gateway HTTP API (v2) Lambda proxy: forwards JSON body to SageMaker runtime."""

from __future__ import annotations

import base64
import json
import os

import boto3
from botocore.exceptions import ClientError

ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
_runtime = boto3.client("sagemaker-runtime")

_JSON_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
}


def lambda_handler(event: dict, context: object) -> dict:
    raw_body = event.get("body")
    if raw_body is None or raw_body == "":
        payload = b"{}"
    elif event.get("isBase64Encoded"):
        payload = base64.b64decode(raw_body)
    else:
        payload = raw_body.encode("utf-8") if isinstance(raw_body, str) else raw_body

    try:
        response = _runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
    except ClientError as e:
        err = e.response.get("Error", {})
        code = err.get("Code", "InvocationError")
        msg = err.get("Message", str(e))
        status = 502 if "ModelError" in code or "InternalFailure" in code else 500
        return {
            "statusCode": status,
            "headers": _JSON_HEADERS,
            "body": json.dumps({"error": code, "message": msg}),
        }

    out_body = response["Body"].read().decode("utf-8")
    return {"statusCode": 200, "headers": _JSON_HEADERS, "body": out_body}
