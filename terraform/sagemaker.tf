resource "aws_sagemaker_model" "classifier" {
  name               = "${var.project_name}-hf-classifier"
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    image          = var.sagemaker_image_uri
    model_data_url = "s3://${aws_s3_bucket.models.bucket}/${aws_s3_object.model_artifact.key}"
  }

  depends_on = [
    aws_s3_object.model_artifact,
    aws_iam_role_policy.sagemaker_inline,
  ]
}

resource "aws_sagemaker_endpoint_configuration" "classifier" {
  name = "${var.project_name}-ep-cfg"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.classifier.name
    initial_variant_weight = 1

    serverless_config {
      memory_size_in_mb = var.sagemaker_serverless_memory_size_in_mb
      max_concurrency   = var.sagemaker_serverless_max_concurrency
    }
  }
}

resource "aws_sagemaker_endpoint" "classifier" {
  name                 = local.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.classifier.name

  depends_on = [
    aws_sagemaker_endpoint_configuration.classifier,
  ]
}

# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

resource "aws_sagemaker_model_package_group" "sentiment" {
  model_package_group_name = var.model_package_group_name
}

resource "aws_sagemaker_pipeline" "training" {
  pipeline_name         = local.pipeline_name
  pipeline_display_name = "${var.project_name}-sentiment-training-pipeline"
  role_arn              = aws_iam_role.sagemaker_pipeline.arn

  pipeline_definition = var.pipeline_definition_json != "" ? var.pipeline_definition_json : file(var.pipeline_definition_path)

  depends_on = [
    aws_iam_role_policy.sagemaker_pipeline_inline,
    aws_sagemaker_model_package_group.sentiment,
  ]
}
