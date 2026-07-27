# The public HTTP API. Each route's Lambda is declared alongside the integration, route
# and invoke permission that expose it.

resource "aws_apigatewayv2_api" "this" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins  = var.cors_allow_origins
    allow_methods  = ["GET", "POST", "OPTIONS"]
    allow_headers  = ["content-type", "authorization"]
    expose_headers = ["x-next-cursor"]
    max_age        = 300
  }
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.this.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_rate_limit  = var.throttle_rate_limit
    throttling_burst_limit = var.throttle_burst_limit
  }
}

locals {
  # The valid-ticker universe can come from SSM; when it does, the readers need
  # GetParameter on exactly that parameter.
  valid_tickers_param_arn = "arn:aws:ssm:${var.aws_region}:${var.aws_account_id}:parameter${var.valid_tickers_ssm_param}"

  read_valid_tickers_statements = var.valid_tickers_ssm_param != "" ? [
    {
      Sid      = "ReadValidTickersParam"
      Effect   = "Allow"
      Action   = ["ssm:GetParameter"]
      Resource = local.valid_tickers_param_arn
    }
  ] : []

  # Env vars shared by the two Lambdas that resolve ticker symbols.
  valid_tickers_env = {
    VALID_TICKERS_SSM_PARAM         = var.valid_tickers_ssm_param
    VALID_TICKERS_JSON              = var.valid_tickers_json
    VALID_TICKERS_FILE              = var.valid_tickers_file
    VALID_TICKERS_CACHE_TTL_SECONDS = tostring(var.valid_tickers_cache_ttl_seconds)
  }
}

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

# ---------------------------------------------------------------------------
# GET /sentiment/cache, GET /sentiment/cache/{symbol}: read-only view of the cache
# ---------------------------------------------------------------------------

module "cache_read" {
  source = "../lambda-function"

  name        = "${var.project_name}-api-cache-read"
  image_uri   = var.image_uris.cache_read
  timeout     = 10
  memory_size = 128

  environment = {
    TABLE_NAME = var.sentiment_cache_table_name
  }

  policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadSentimentCache"
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:Scan",
        ]
        Resource = var.sentiment_cache_table_arn
      },
    ]
  })
}

resource "aws_apigatewayv2_integration" "cache_read" {
  api_id                 = aws_apigatewayv2_api.this.id
  integration_type       = "AWS_PROXY"
  integration_uri        = module.cache_read.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "get_sentiment_cache_all" {
  api_id    = aws_apigatewayv2_api.this.id
  route_key = "GET /sentiment/cache"
  target    = "integrations/${aws_apigatewayv2_integration.cache_read.id}"
}

resource "aws_apigatewayv2_route" "get_sentiment_cache" {
  api_id    = aws_apigatewayv2_api.this.id
  route_key = "GET /sentiment/cache/{symbol}"
  target    = "integrations/${aws_apigatewayv2_integration.cache_read.id}"
}

resource "aws_lambda_permission" "cache_read" {
  statement_id  = "AllowAPIGatewayInvokeCacheRead"
  action        = "lambda:InvokeFunction"
  function_name = module.cache_read.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.this.execution_arn}/*/*"
}

# ---------------------------------------------------------------------------
# GET /tickers/suggest: autocomplete over the valid-ticker universe
# ---------------------------------------------------------------------------

module "ticker_suggest" {
  source = "../lambda-function"

  name        = "${var.project_name}-api-ticker-suggest"
  image_uri   = var.image_uris.ticker_suggest
  timeout     = 10
  memory_size = 128

  environment = local.valid_tickers_env

  # Reading the ticker universe from the packaged file needs no permissions at all,
  # so without an SSM parameter this function gets logs access and nothing else.
  policy_json = length(local.read_valid_tickers_statements) > 0 ? jsonencode({
    Version   = "2012-10-17"
    Statement = local.read_valid_tickers_statements
  }) : null
}

resource "aws_apigatewayv2_integration" "ticker_suggest" {
  api_id                 = aws_apigatewayv2_api.this.id
  integration_type       = "AWS_PROXY"
  integration_uri        = module.ticker_suggest.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "get_ticker_suggestions" {
  api_id    = aws_apigatewayv2_api.this.id
  route_key = "GET /tickers/suggest"
  target    = "integrations/${aws_apigatewayv2_integration.ticker_suggest.id}"
}

resource "aws_lambda_permission" "ticker_suggest" {
  statement_id  = "AllowAPIGatewayInvokeTickerSuggest"
  action        = "lambda:InvokeFunction"
  function_name = module.ticker_suggest.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.this.execution_arn}/*/*"
}
