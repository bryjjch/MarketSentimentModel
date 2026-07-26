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
