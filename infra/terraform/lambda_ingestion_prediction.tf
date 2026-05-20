resource "aws_lambda_function" "ingestion_prediction" {
  function_name = "${var.project_name}-ingestion-prediction"
  role          = aws_iam_role.ingestion_prediction_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.ingestion_prediction.repository_url}:latest"

  timeout     = 600
  memory_size = var.prediction_lambda_memory_mb

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME    = aws_sagemaker_endpoint.classifier.name
      DATA_BUCKET                = aws_s3_bucket.data.bucket
      PSEUDO_LABEL_FUNCTION_NAME = aws_lambda_function.pseudo_label.function_name
      CACHE_TABLE_NAME           = aws_dynamodb_table.sentiment_cache.name
      CACHE_TTL_SECONDS          = tostring(var.sentiment_cache_ttl_seconds)
      RECENT_HEADLINES_MAX       = "10"
      LOW_CONF_TOP_PROB          = tostring(var.low_conf_top_prob)
      LOW_CONF_MARGIN            = tostring(var.low_conf_margin)
      SAGEMAKER_BATCH_SIZE       = tostring(var.sagemaker_batch_size)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.ingestion_prediction_lambda_basic,
    aws_iam_role_policy.ingestion_prediction_lambda,
    aws_s3_bucket.data,
    aws_sagemaker_endpoint.classifier,
    aws_lambda_function.pseudo_label,
    aws_dynamodb_table.sentiment_cache,
  ]
}
