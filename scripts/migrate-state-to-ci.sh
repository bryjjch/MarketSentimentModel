#!/usr/bin/env bash
# One-time migration: stop Terraform managing the resources that CI and the
# model_promote Lambda now own.
#
# Run this ONCE, from a workstation with admin credentials, BEFORE the first
# `terraform apply` on the new configuration. Without it, apply destroys a live
# serving endpoint and the training pipeline:
#
#   aws_sagemaker_model.classifier                  -> owned by model_promote
#   aws_sagemaker_endpoint_configuration.classifier -> owned by model_promote
#   aws_sagemaker_endpoint.classifier               -> owned by model_promote
#   aws_s3_object.model_artifact                    -> written by SageMaker training jobs
#   aws_sagemaker_pipeline.training                 -> upserted by CI
#
# `terraform state rm` only forgets a resource. Nothing in AWS is touched: the
# endpoint keeps serving traffic throughout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/terraform"

RESOURCES=(
  aws_sagemaker_model.classifier
  aws_sagemaker_endpoint_configuration.classifier
  aws_sagemaker_endpoint.classifier
  aws_s3_object.model_artifact
  aws_sagemaker_pipeline.training
)

if [ ! -d .terraform ]; then
  echo "Running terraform init..."
  terraform init -input=false -backend-config=backend.hcl
fi

echo "State entries to remove:"
present=()
for r in "${RESOURCES[@]}"; do
  if terraform state list "$r" >/dev/null 2>&1; then
    echo "  $r"
    present+=("$r")
  else
    echo "  $r (already absent, skipping)"
  fi
done

if [ ${#present[@]} -eq 0 ]; then
  echo ""
  echo "Nothing to do; migration has already run."
  exit 0
fi

echo ""
echo "These resources will be REMOVED FROM STATE ONLY. AWS is not modified."
read -r -p "Proceed? [y/N] " reply
case "$reply" in
  [yY]) ;;
  *) echo "Aborted."; exit 1 ;;
esac

for r in "${present[@]}"; do
  terraform state rm "$r"
done

cat <<'EOF'

Done. Next steps:

  1. Verify the plan no longer destroys anything:

       terraform plan -var-file=env/prod.tfvars \
         -var "image_tag=$(../scripts/image-tag.sh)"

     Expect creates and in-place updates only. A destroy here means something was
     missed — stop and investigate rather than applying.

  2. Build and push the new model_promote image, which the apply needs:

       ../scripts/lambda-build-push.sh

  3. Apply, from this workstation for the bootstrap only (it creates the GitHub
     OIDC provider and the CI roles that every later apply uses):

       terraform apply -var-file=env/prod.tfvars \
         -var "image_tag=$(../scripts/image-tag.sh)"

  4. Set the repository variables GitHub Actions needs:

       gh variable set AWS_REGION          --body "$(terraform output -raw aws_region 2>/dev/null || echo us-east-1)"
       gh variable set AWS_PLAN_ROLE_ARN   --body "$(terraform output -raw github_plan_role_arn)"
       gh variable set AWS_APPLY_ROLE_ARN  --body "$(terraform output -raw github_apply_role_arn)"
       gh variable set TF_VAR_GOOGLE_SECRET_ARN --body "<your Secrets Manager ARN>"

  5. Upsert the training pipeline once, so the retrain rule has a target:

       PYTHONPATH=src/sagemaker python -m pipeline.build_pipeline \
         --role "$(terraform output -raw pipeline_role_arn)" \
         --pipeline-name "$(terraform output -raw pipeline_name)" \
         --bucket "$(terraform output -raw data_bucket_name)" \
         --upsert

  From here on, pushing to main does all of the above.
EOF
