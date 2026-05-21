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
# Routes: /predict
# ---------------------------------------------------------------------------

resource "aws_apigatewayv2_integration" "api_inference" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api_inference.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "post_predict" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.api_inference.id}"
}

resource "aws_lambda_permission" "apigw_invoke_api_inference" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api_inference.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

# ---------------------------------------------------------------------------
# Routes: /sentiment/by-symbol, /sentiment/cache, /tickers/suggest
# ---------------------------------------------------------------------------

resource "aws_apigatewayv2_integration" "api_sentiment_by_symbol" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api_sentiment_by_symbol.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "post_sentiment_by_symbol" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "POST /sentiment/by-symbol"
  target    = "integrations/${aws_apigatewayv2_integration.api_sentiment_by_symbol.id}"
}

resource "aws_lambda_permission" "apigw_invoke_api_sentiment_by_symbol" {
  statement_id  = "AllowAPIGatewayInvokeSentiment"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api_sentiment_by_symbol.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

resource "aws_apigatewayv2_integration" "cache_read" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.cache_read.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "get_sentiment_cache_all" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /sentiment/cache"
  target    = "integrations/${aws_apigatewayv2_integration.cache_read.id}"
}

resource "aws_apigatewayv2_route" "get_sentiment_cache" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /sentiment/cache/{symbol}"
  target    = "integrations/${aws_apigatewayv2_integration.cache_read.id}"
}

resource "aws_apigatewayv2_route" "get_ticker_suggestions" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /tickers/suggest"
  target    = "integrations/${aws_apigatewayv2_integration.cache_read.id}"
}

resource "aws_lambda_permission" "apigw_invoke_cache_read" {
  statement_id  = "AllowAPIGatewayInvokeCacheRead"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.cache_read.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}
