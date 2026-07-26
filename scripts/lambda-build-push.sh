#!/usr/bin/env bash
# Build and push all Lambda container images to ECR.
# Usage: ./scripts/lambda-build-push.sh [function_name ...]
# If no function names are given, builds all nine.
# Requires: docker, aws CLI, terraform output available in terraform/.
#
# CI does this too (.github/workflows/deploy.yml). This script stays for local
# iteration; both derive the same tag from scripts/image-tag.sh, so an image built
# here and an image built by CI from the same commit are interchangeable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TERRAFORM_DIR="$REPO_ROOT/terraform"
LAMBDAS_DIR="$REPO_ROOT/src/lambdas"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region 2>/dev/null || echo "${AWS_DEFAULT_REGION:-us-east-1}")
PROJECT=$(cd "$TERRAFORM_DIR" && terraform output -raw project_name 2>/dev/null || echo "finsense")

REPO_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_TAG="$("$REPO_ROOT/scripts/image-tag.sh")"
TFVARS_FILE="$TERRAFORM_DIR/image_tag.auto.tfvars"

echo "Checking Lambda sources compile..."
python3 "$REPO_ROOT/scripts/check_lambda_sources.py" "$LAMBDAS_DIR"

# The tag is the committed tree's hash, so uncommitted edits produce an image whose
# tag describes different contents than what was built.
if ! git -C "$REPO_ROOT" diff --quiet HEAD -- "$LAMBDAS_DIR"; then
  echo "WARNING: uncommitted changes under src/lambdas; image tag will not match its contents." >&2
fi

echo "Logging in to ECR ($REPO_BASE)..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "$REPO_BASE"

# Map function directory name to ECR repo suffix (dir uses underscores, repo uses hyphens)
ALL_FUNCS=(api_cache_read api_sentiment api_ticker_suggest cache_write model_promote pipeline_collect pipeline_dispatch pipeline_label pipeline_predict)
FUNCS=("${@:-${ALL_FUNCS[@]}}")

for func in "${FUNCS[@]}"; do
  repo_name="${PROJECT}-${func//_/-}"
  image="${REPO_BASE}/${repo_name}:${IMAGE_TAG}"
  echo ""

  # ECR repos are IMMUTABLE, so re-pushing an existing tag is a hard error. The tag
  # only changes when src/lambdas changes, so an existing tag means this exact source
  # is already published.
  if aws ecr describe-images --region "$REGION" --repository-name "$repo_name" \
       --image-ids "imageTag=${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "==> Skipping $func: ${repo_name}:${IMAGE_TAG} already in ECR"
    continue
  fi

  echo "==> Building $func → $image"
  docker build \
    --platform linux/amd64 \
    --provenance=false \
    -f "$LAMBDAS_DIR/$func/Dockerfile" \
    -t "$image" \
    "$LAMBDAS_DIR"
  echo "==> Pushing $func"
  docker push "$image"
done

echo ""
echo "image_tag = \"${IMAGE_TAG}\"" > "$TFVARS_FILE"
echo "Wrote image tag to $TFVARS_FILE"
echo "Run 'terraform apply' in terraform/ to update Lambda image URIs."
