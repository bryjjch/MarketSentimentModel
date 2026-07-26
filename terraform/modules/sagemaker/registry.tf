# ---------------------------------------------------------------------------
# Model registry
#
# The model, endpoint configuration and endpoint are NOT Terraform resources. They
# are created and rolled forward by the model_promote Lambda (see promotion.tf) each
# time a model package is approved. Describing them here would mean:
#   - every deploy needed the 405 MB model.tar.gz on the machine running apply, and
#   - every retrain showed up as drift Terraform wanted to revert.
#
# Terraform still owns the endpoint's *name*, its execution role, and its serverless
# sizing, which reach the promoter as env vars.
#
# Migrating an existing deployment: drop them from state without touching AWS —
#   terraform state rm aws_sagemaker_model.classifier \
#                      aws_sagemaker_endpoint_configuration.classifier \
#                      aws_sagemaker_endpoint.classifier \
#                      aws_s3_object.model_artifact
#
# The Pipeline itself is likewise not a Terraform resource. Its definition is produced
# by the SageMaker SDK (src/sagemaker/pipeline/build_pipeline.py), which uploads
# sourcedir.tar.gz to S3 while compiling and bakes the pipeline role's ARN into the
# JSON. Managing it here meant Terraform needed a generated file that could only be
# generated after Terraform had run. CI breaks the cycle by applying Terraform first,
# then running `build_pipeline.py --upsert` with the role ARN from an output.
# See the sagemaker-pipeline job in .github/workflows/deploy.yml.
# ---------------------------------------------------------------------------

resource "aws_sagemaker_model_package_group" "this" {
  model_package_group_name = var.model_package_group_name
}
