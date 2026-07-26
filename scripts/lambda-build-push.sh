#!/usr/bin/env bash
# Build and push all Lambda container images to ECR.
# Usage: ./scripts/lambda-build-push.sh [function_name ...]
# If no function names are given, builds all eight.
# Requires: docker, aws CLI, terraform output available in terraform/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TERRAFORM_DIR="$REPO_ROOT/terraform"
LAMBDAS_DIR="$REPO_ROOT/src/lambdas"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region 2>/dev/null || echo "${AWS_DEFAULT_REGION:-us-east-1}")
PROJECT=$(cd "$TERRAFORM_DIR" && terraform output -raw project_name 2>/dev/null || echo "finsense")

REPO_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_TAG="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
TFVARS_FILE="$REPO_ROOT/terraform/image_tag.auto.tfvars"

# Docker COPY is byte-for-byte, so a source file Python cannot parse (e.g. text saved
# as cp1252 instead of UTF-8) builds and pushes cleanly, then fails at Lambda import
# time with Runtime.UserCodeSyntaxError. Catch it here instead.
echo "Checking Lambda sources compile..."
python3 - "$LAMBDAS_DIR" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
bad = []
for path in sorted(root.rglob("*.py")):
    if "__pycache__" in path.parts:
        continue
    try:
        compile(path.read_bytes(), str(path), "exec")
    except (SyntaxError, UnicodeDecodeError) as exc:
        bad.append(f"{path}: {exc}")

if bad:
    print("ERROR: Lambda sources failed to compile; refusing to build:", file=sys.stderr)
    for line in bad:
        print(f"  {line}", file=sys.stderr)
    sys.exit(1)
PY

# The image is tagged with HEAD's SHA, so uncommitted edits produce an image whose tag
# does not describe its contents.
if ! git -C "$REPO_ROOT" diff --quiet HEAD -- "$LAMBDAS_DIR"; then
  echo "WARNING: uncommitted changes under src/lambdas; image tag will not match its contents." >&2
fi

echo "Logging in to ECR ($REPO_BASE)..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "$REPO_BASE"

# Map function directory name to ECR repo suffix (dir uses underscores, repo uses hyphens)
ALL_FUNCS=(api_cache_read api_sentiment api_ticker_suggest cache_write pipeline_collect pipeline_dispatch pipeline_label pipeline_predict)
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
echo "Run 'terraform apply' in terraform/ to update Lambda image URIs."
