# The public HTTP API. Each route's Lambda lives in its own file alongside the
# integration, route and invoke permission that expose it.

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
