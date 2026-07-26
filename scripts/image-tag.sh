#!/usr/bin/env bash
# Print the ECR image tag for the current commit's Lambda sources.
#
# The tag is the git tree hash of src/lambdas rather than the commit SHA. That means
# it changes if and only if Lambda source, requirements or Dockerfiles change, so:
#
#   - a Terraform-only or frontend-only commit reuses the existing tag, and both the
#     ECR skip-check and `terraform plan` correctly show nothing to do;
#   - the tag is derivable offline from any checkout, so CI and local builds agree
#     without passing state through a file or a parameter store.
#
# Caveat: a moving base image (public.ecr.aws/lambda/python:3.12) does not change the
# tree, so it will not trigger a rebuild. Force one with the rebuild_images input on
# the deploy workflow, or by pushing a Dockerfile change.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
git -C "$REPO_ROOT" rev-parse --short "HEAD:src/lambdas"
