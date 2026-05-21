resource "aws_cloudwatch_event_rule" "ingestion" {
  name                = "${var.project_name}-ingestion"
  description         = "Daily fan-out that triggers the ingestion Lambda; ingestion then fans out to the prediction Lambda per ticker."
  schedule_expression = var.ingestion_schedule
}

resource "aws_cloudwatch_event_target" "ingestion" {
  rule      = aws_cloudwatch_event_rule.ingestion.name
  target_id = "IngestionLambda"
  arn       = aws_lambda_function.ingestion.arn
}

resource "aws_lambda_permission" "eventbridge_invoke_ingestion" {
  statement_id  = "AllowEventBridgeInvokeIngestion"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingestion.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.ingestion.arn
}
