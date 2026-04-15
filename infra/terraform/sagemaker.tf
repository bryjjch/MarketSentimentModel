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
    initial_instance_count = 1
    instance_type          = var.sagemaker_instance_type
    initial_variant_weight = 1
  }
}

resource "aws_sagemaker_endpoint" "classifier" {
  name                 = local.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.classifier.name

  depends_on = [
    aws_sagemaker_endpoint_configuration.classifier,
  ]
}
