data "archive_file" "sentiment_refresh_lambda" {
  count       = var.disable_legacy_sentiment_refresh ? 0 : 1
  type        = "zip"
  source_file = abspath("${path.module}/../lambda/sentiment_refresh/handler.py")
  output_path = "${path.module}/build/sentiment_refresh.zip"
}

resource "aws_lambda_function" "sentiment_refresh" {
  count         = var.disable_legacy_sentiment_refresh ? 0 : 1
  function_name = "${var.project_name}-sentiment-refresh"
  role          = aws_iam_role.sentiment_refresh_lambda[0].arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.12"

  filename         = data.archive_file.sentiment_refresh_lambda[0].output_path
  source_code_hash = data.archive_file.sentiment_refresh_lambda[0].output_base64sha256

  timeout     = 300
  memory_size = 256

  environment {
    variables = {
      SENTIMENT_FUNCTION_NAME = aws_lambda_function.sentiment.function_name
      TABLE_NAME              = aws_dynamodb_table.sentiment_cache.name
      CACHE_TTL_SECONDS       = tostring(var.sentiment_cache_ttl_seconds)
      TOP_TICKERS_SSM_PARAM   = aws_ssm_parameter.top_tickers.name
      DEFAULT_TICKERS_JSON    = var.top_tickers_json
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.sentiment_refresh_lambda_basic,
    aws_iam_role_policy.sentiment_refresh_invoke,
    aws_lambda_function.sentiment,
    aws_dynamodb_table.sentiment_cache,
    aws_ssm_parameter.top_tickers,
  ]
}
