data "archive_file" "ingestion_lambda" {
  type        = "zip"
  source_dir  = abspath("${path.module}/../lambda/ingestion")
  output_path = "${path.module}/build/ingestion.zip"
  excludes    = ["__pycache__", "*.pyc", ".pytest_cache"]
}

resource "aws_lambda_function" "ingestion" {
  function_name = "${var.project_name}-ingestion"
  role          = aws_iam_role.ingestion_lambda.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.12"

  filename         = data.archive_file.ingestion_lambda.output_path
  source_code_hash = data.archive_file.ingestion_lambda.output_base64sha256

  layers = [
    aws_lambda_layer_version.finsense_shared.arn,
    aws_lambda_layer_version.finsense_deps.arn,
  ]

  timeout     = 600
  memory_size = var.ingestion_lambda_memory_mb

  environment {
    variables = {
      DATA_BUCKET                        = aws_s3_bucket.data.bucket
      INGESTION_PREDICTION_FUNCTION_NAME = aws_lambda_function.ingestion_prediction.function_name
      DEFAULT_MAX_ARTICLES               = tostring(var.ingestion_max_articles)
      INCLUDE_SOCIAL                     = var.ingestion_include_social ? "true" : "false"
      TOP_TICKERS_SSM_PARAM              = aws_ssm_parameter.top_tickers.name
      DEFAULT_TICKERS_JSON               = var.top_tickers_json
      REDDIT_SECRET_ARN                  = var.reddit_credentials_secret_arn
      FINNHUB_SECRET_ARN               = var.finnhub_secret_arn
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.ingestion_lambda_basic,
    aws_iam_role_policy.ingestion_lambda,
    aws_s3_bucket.data,
    aws_lambda_function.ingestion_prediction,
    aws_ssm_parameter.top_tickers,
  ]
}
