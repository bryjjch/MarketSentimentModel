# ---------------------------------------------------------------------------
# POST /sentiment/by-symbol: real-time per-symbol sentiment
# ---------------------------------------------------------------------------

module "sentiment" {
  source = "../lambda-function"

  name      = "${var.project_name}-api-sentiment"
  image_uri = var.image_uris.sentiment
  # One second under API Gateway's 30s integration limit, so the client sees the
  # Lambda's own error rather than a gateway timeout.
  timeout     = 29
  memory_size = 512

  environment = merge(local.valid_tickers_env, {
    SAGEMAKER_ENDPOINT_NAME = var.sagemaker_endpoint_name
    REDDIT_SECRET_ARN       = var.reddit_credentials_secret_arn
    RECENT_HEADLINES_MAX    = "10"
    DEFAULT_MAX_ARTICLES    = "12"
    CACHE_WRITE_QUEUE_URL   = var.cache_write_queue_url
    CACHE_TTL_SECONDS       = tostring(var.sentiment_cache_api_ttl_seconds)
    RSS_OVERFETCH           = tostring(var.rss_overfetch)
  })

  policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid      = "InvokeSageMakerEndpoint"
          Effect   = "Allow"
          Action   = ["sagemaker:InvokeEndpoint"]
          Resource = var.sagemaker_endpoint_arn
        },
        {
          Sid      = "SendCacheWriteTasks"
          Effect   = "Allow"
          Action   = ["sqs:SendMessage"]
          Resource = var.cache_write_queue_arn
        },
      ],
      var.reddit_credentials_secret_arn != "" ? [
        {
          Sid      = "ReadRedditSecret"
          Effect   = "Allow"
          Action   = ["secretsmanager:GetSecretValue"]
          Resource = var.reddit_credentials_secret_arn
        }
      ] : [],
      local.read_valid_tickers_statements
    )
  })
}

resource "aws_apigatewayv2_integration" "sentiment" {
  api_id                 = aws_apigatewayv2_api.this.id
  integration_type       = "AWS_PROXY"
  integration_uri        = module.sentiment.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "post_sentiment_by_symbol" {
  api_id    = aws_apigatewayv2_api.this.id
  route_key = "POST /sentiment/by-symbol"
  target    = "integrations/${aws_apigatewayv2_integration.sentiment.id}"
}

resource "aws_lambda_permission" "sentiment" {
  statement_id  = "AllowAPIGatewayInvokeSentiment"
  action        = "lambda:InvokeFunction"
  function_name = module.sentiment.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.this.execution_arn}/*/*"
}
