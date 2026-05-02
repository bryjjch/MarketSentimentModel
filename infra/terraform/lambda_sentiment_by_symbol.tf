data "archive_file" "api_sentiment_by_symbol_lambda" {
  type        = "zip"
  source_dir  = abspath("${path.module}/../lambda/api_sentiment_by_symbol")
  output_path = "${path.module}/build/api_sentiment_by_symbol.zip"
  excludes    = ["__pycache__", "*.pyc", ".pytest_cache"]
}

resource "aws_lambda_function" "api_sentiment_by_symbol" {
  function_name = "${var.project_name}-api-sentiment-by-symbol"
  role          = aws_iam_role.api_sentiment_by_symbol_lambda.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.12"

  filename         = data.archive_file.api_sentiment_by_symbol_lambda.output_path
  source_code_hash = data.archive_file.api_sentiment_by_symbol_lambda.output_base64sha256
  layers = [
    aws_lambda_layer_version.finsense_shared.arn,
    aws_lambda_layer_version.finsense_deps.arn,
  ]

  timeout     = 29
  memory_size = 512

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = aws_sagemaker_endpoint.classifier.name
      REDDIT_SECRET_ARN       = var.reddit_credentials_secret_arn
      RECENT_HEADLINES_MAX    = "10"
      DEFAULT_MAX_ARTICLES    = "12"
      CACHE_TABLE_NAME        = aws_dynamodb_table.sentiment_cache.name
      CACHE_TTL_SECONDS       = tostring(var.sentiment_cache_api_ttl_seconds)
      VALID_TICKERS_SSM_PARAM = var.valid_tickers_ssm_param
      VALID_TICKERS_JSON      = var.valid_tickers_json
      VALID_TICKERS_FILE      = var.valid_tickers_file
      VALID_TICKERS_CACHE_TTL_SECONDS = tostring(var.valid_tickers_cache_ttl_seconds)
      RSS_OVERFETCH                   = tostring(var.rss_overfetch)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.api_sentiment_by_symbol_lambda_basic,
    aws_iam_role_policy.api_sentiment_by_symbol_lambda_invoke,
    aws_iam_role_policy.api_sentiment_by_symbol_lambda_cache_write,
    aws_dynamodb_table.sentiment_cache,
    aws_sagemaker_endpoint.classifier,
  ]
}
