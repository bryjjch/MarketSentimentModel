resource "aws_cloudwatch_event_rule" "sentiment_refresh" {
  count               = var.disable_legacy_sentiment_refresh ? 0 : 1
  name                = "${var.project_name}-sentiment-refresh"
  description         = "Legacy: trigger sentiment cache refresh on a schedule (superseded by the ingestion pipeline)"
  schedule_expression = var.sentiment_refresh_schedule
}

resource "aws_cloudwatch_event_target" "sentiment_refresh" {
  count     = var.disable_legacy_sentiment_refresh ? 0 : 1
  rule      = aws_cloudwatch_event_rule.sentiment_refresh[0].name
  target_id = "SentimentRefreshLambda"
  arn       = aws_lambda_function.sentiment_refresh[0].arn
}

resource "aws_lambda_permission" "eventbridge_invoke_sentiment_refresh" {
  count         = var.disable_legacy_sentiment_refresh ? 0 : 1
  statement_id  = "AllowEventBridgeInvokeRefresh"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sentiment_refresh[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.sentiment_refresh[0].arn
}
