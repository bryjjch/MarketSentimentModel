resource "aws_cloudwatch_event_rule" "sentiment_refresh" {
  name                = "${var.project_name}-sentiment-refresh"
  description         = "Trigger sentiment cache refresh on a schedule"
  schedule_expression = var.sentiment_refresh_schedule
}

resource "aws_cloudwatch_event_target" "sentiment_refresh" {
  rule      = aws_cloudwatch_event_rule.sentiment_refresh.name
  target_id = "SentimentRefreshLambda"
  arn       = aws_lambda_function.sentiment_refresh.arn
}

resource "aws_lambda_permission" "eventbridge_invoke_sentiment_refresh" {
  statement_id  = "AllowEventBridgeInvokeRefresh"
  action          = "lambda:InvokeFunction"
  function_name   = aws_lambda_function.sentiment_refresh.function_name
  principal       = "events.amazonaws.com"
  source_arn      = aws_cloudwatch_event_rule.sentiment_refresh.arn
}
