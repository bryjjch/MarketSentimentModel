resource "aws_apigatewayv2_integration" "sentiment" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri          = aws_lambda_function.sentiment.invoke_arn
  integration_method     = "POST"
  payload_format_version   = "2.0"
}

resource "aws_apigatewayv2_route" "post_sentiment_by_symbol" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "POST /sentiment/by-symbol"
  target    = "integrations/${aws_apigatewayv2_integration.sentiment.id}"
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

resource "aws_lambda_permission" "apigw_invoke_sentiment" {
  statement_id  = "AllowAPIGatewayInvokeSentiment"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sentiment.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

resource "aws_lambda_permission" "apigw_invoke_cache_read" {
  statement_id  = "AllowAPIGatewayInvokeCacheRead"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.cache_read.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}
