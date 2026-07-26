# ---------------------------------------------------------------------------
# dispatch: cron entrypoint; enumerates tickers and enqueues one collect task each
# ---------------------------------------------------------------------------

resource "aws_ssm_parameter" "top_tickers" {
  name  = "/${var.project_name}/top-tickers"
  type  = "String"
  value = var.top_tickers_json

  description = "JSON array of tickers for sentiment cache refresher (edit in console or terraform)."
}

module "dispatch" {
  source = "../lambda-function"

  name        = "${var.project_name}-pipeline-dispatch"
  image_uri   = var.image_uris.dispatch
  timeout     = 60
  memory_size = 128

  environment = {
    COLLECT_QUEUE_URL     = var.queue_urls.collect
    DEFAULT_MAX_ARTICLES  = tostring(var.ingestion_max_articles)
    INCLUDE_SOCIAL        = var.ingestion_include_social ? "true" : "false"
    TOP_TICKERS_SSM_PARAM = aws_ssm_parameter.top_tickers.name
    DEFAULT_TICKERS_JSON  = var.top_tickers_json
  }

  policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadTickerParam"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = aws_ssm_parameter.top_tickers.arn
      },
      {
        Sid      = "SendCollectTasks"
        Effect   = "Allow"
        Action   = ["sqs:SendMessage"]
        Resource = var.queue_arns.collect
      },
    ]
  })
}

# --- Daily trigger -----------------------------------------------------------

resource "aws_cloudwatch_event_rule" "dispatch" {
  name                = "${var.project_name}-pipeline-dispatch"
  description         = "Daily trigger for the dispatch Lambda, which enqueues one collect task per ticker."
  schedule_expression = var.ingestion_schedule
}

resource "aws_cloudwatch_event_target" "dispatch" {
  rule      = aws_cloudwatch_event_rule.dispatch.name
  target_id = "PipelineDispatchLambda"
  arn       = module.dispatch.arn
}

resource "aws_lambda_permission" "dispatch_events" {
  statement_id  = "AllowEventBridgeInvokePipelineDispatch"
  action        = "lambda:InvokeFunction"
  function_name = module.dispatch.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.dispatch.arn
}
