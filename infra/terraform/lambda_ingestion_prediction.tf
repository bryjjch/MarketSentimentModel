data "archive_file" "ingestion_prediction_lambda" {
  type        = "zip"
  source_dir  = abspath("${path.module}/../lambda/ingestion_prediction")
  output_path = "${path.module}/build/ingestion_prediction.zip"
  excludes    = ["__pycache__", "*.pyc", ".pytest_cache"]
}

resource "aws_lambda_function" "ingestion_prediction" {
  function_name = "${var.project_name}-ingestion-prediction"
  role          = aws_iam_role.ingestion_prediction_lambda.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.12"

  filename         = data.archive_file.ingestion_prediction_lambda.output_path
  source_code_hash = data.archive_file.ingestion_prediction_lambda.output_base64sha256

  layers = [
    aws_lambda_layer_version.finsense_shared.arn,
    aws_lambda_layer_version.finsense_deps.arn,
  ]

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
