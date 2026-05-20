#!/usr/bin/env python3
"""Generate the FinSense SageMaker Pipeline definition JSON.

Usage
-----
Generate JSON for Terraform deployment (offline, no AWS calls):

    PYTHONPATH=src python -m sagemaker.pipeline.build_pipeline \\
        --role arn:aws:iam::123456789012:role/SageMakerPipelineRole \\
        --output terraform/pipeline_definition.json

Upsert the pipeline directly to SageMaker:

    PYTHONPATH=src python -m sagemaker.pipeline.build_pipeline \\
        --role arn:aws:iam::123456789012:role/SageMakerPipelineRole \\
        --upsert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession


def _offline_session(region: str, default_bucket: str) -> PipelineSession:
    """Create a session for JSON generation."""
    import boto3

    boto_session = boto3.Session(region_name=region)
    return PipelineSession(
        boto_session=boto_session,
        default_bucket=default_bucket,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build or upsert the FinSense SageMaker Pipeline")
    p.add_argument(
        "--role",
        required=True,
        help="SageMaker execution role ARN",
    )
    p.add_argument(
        "--pipeline-name",
        default="FinSenseSentimentPipeline",
        help="Pipeline name (default: FinSenseSentimentPipeline)",
    )
    p.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    p.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket for pipeline artifacts (default: SageMaker session default)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write pipeline JSON to this file (for Terraform). Omit to print to stdout.",
    )
    p.add_argument(
        "--upsert",
        action="store_true",
        help="Create or update the pipeline in SageMaker (requires AWS credentials).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from .pipeline_definition import build_pipeline

    bucket = args.bucket or f"sagemaker-{args.region}-pipeline"

    session = _offline_session(args.region, bucket)

    pipeline: Pipeline = build_pipeline(
        role=args.role,
        pipeline_name=args.pipeline_name,
        default_bucket=bucket,
        sagemaker_session=session,
    )

    if args.upsert:
        pipeline.upsert(role_arn=args.role)
        print(f"Pipeline '{args.pipeline_name}' upserted successfully.", file=sys.stderr)
        return

    definition = json.loads(pipeline.definition())
    pretty = json.dumps(definition, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n", encoding="utf-8")
        print(f"Wrote pipeline definition to {args.output}", file=sys.stderr)
    else:
        print(pretty)


if __name__ == "__main__":
    main()
