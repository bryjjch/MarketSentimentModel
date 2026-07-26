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
