#!/usr/bin/env bash
# Build and push all Lambda container images to ECR.
# Usage: ./scripts/lambda-build-push.sh [function_name ...]
# If no function names are given, builds all six.
# Requires: docker, aws CLI, terraform output available in infra/terraform/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TERRAFORM_DIR="$REPO_ROOT/infra/terraform"
LAMBDAS_DIR="$REPO_ROOT/src/lambdas"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region 2>/dev/null || echo "${AWS_DEFAULT_REGION:-us-east-1}")
PROJECT=$(cd "$TERRAFORM_DIR" && terraform output -raw project_name 2>/dev/null || echo "finsense")

REPO_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_TAG="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
TFVARS_FILE="$REPO_ROOT/infra/terraform/image_tag.auto.tfvars"

echo "Logging in to ECR ($REPO_BASE)..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "$REPO_BASE"

# Map function directory name to ECR repo suffix (dir uses underscores, repo uses hyphens)
ALL_FUNCS=(api_inference api_sentiment_by_symbol cache_read ingestion ingestion_prediction pseudo_label)
FUNCS=("${@:-${ALL_FUNCS[@]}}")

for func in "${FUNCS[@]}"; do
  repo_name="${PROJECT}-${func//_/-}"
  image="${REPO_BASE}/${repo_name}:${IMAGE_TAG}"
  echo ""
  echo "==> Building $func → $image"
  docker build \
    --platform linux/amd64 \
    --provenance=false \
    --no-cache \
    -f "$LAMBDAS_DIR/$func/Dockerfile" \
    -t "$image" \
    "$LAMBDAS_DIR"
  echo "==> Pushing $func"
  docker push "$image"
done

echo ""
echo "image_tag = \"${IMAGE_TAG}\"" > "$TFVARS_FILE"
echo "Wrote image tag to $TFVARS_FILE"
echo "Run 'terraform apply' in infra/terraform/ to update Lambda image URIs."
