resource "aws_apigatewayv2_api" "http" {
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
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_rate_limit  = var.apigateway_throttle_rate_limit
    throttling_burst_limit = var.apigateway_throttle_burst_limit
  }
}

# ---------------------------------------------------------------------------
# Routes: /sentiment/by-symbol
# ---------------------------------------------------------------------------

resource "aws_apigatewayv2_integration" "api_sentiment" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api_sentiment.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "post_sentiment_by_symbol" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "POST /sentiment/by-symbol"
  target    = "integrations/${aws_apigatewayv2_integration.api_sentiment.id}"
}

resource "aws_lambda_permission" "apigw_invoke_api_sentiment" {
  statement_id  = "AllowAPIGatewayInvokeSentiment"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api_sentiment.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

# ---------------------------------------------------------------------------
# Routes: /sentiment/cache, /sentiment/cache/{symbol}
# ---------------------------------------------------------------------------

resource "aws_apigatewayv2_integration" "api_cache_read" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api_cache_read.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "get_sentiment_cache_all" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /sentiment/cache"
  target    = "integrations/${aws_apigatewayv2_integration.api_cache_read.id}"
}

resource "aws_apigatewayv2_route" "get_sentiment_cache" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /sentiment/cache/{symbol}"
  target    = "integrations/${aws_apigatewayv2_integration.api_cache_read.id}"
}

resource "aws_lambda_permission" "apigw_invoke_api_cache_read" {
  statement_id  = "AllowAPIGatewayInvokeCacheRead"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api_cache_read.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

# ---------------------------------------------------------------------------
# Routes: /tickers/suggest
# ---------------------------------------------------------------------------

resource "aws_apigatewayv2_integration" "api_ticker_suggest" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api_ticker_suggest.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "get_ticker_suggestions" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /tickers/suggest"
  target    = "integrations/${aws_apigatewayv2_integration.api_ticker_suggest.id}"
}

resource "aws_lambda_permission" "apigw_invoke_api_ticker_suggest" {
  statement_id  = "AllowAPIGatewayInvokeTickerSuggest"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api_ticker_suggest.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}
