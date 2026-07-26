resource "aws_cloudwatch_event_rule" "pipeline_dispatch" {
  name                = "${var.project_name}-pipeline-dispatch"
  description         = "Daily trigger for the dispatch Lambda, which enqueues one collect task per ticker."
  schedule_expression = var.ingestion_schedule
}

resource "aws_cloudwatch_event_target" "pipeline_dispatch" {
  rule      = aws_cloudwatch_event_rule.pipeline_dispatch.name
  target_id = "PipelineDispatchLambda"
  arn       = aws_lambda_function.pipeline_dispatch.arn
}

resource "aws_lambda_permission" "eventbridge_invoke_pipeline_dispatch" {
  statement_id  = "AllowEventBridgeInvokePipelineDispatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pipeline_dispatch.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.pipeline_dispatch.arn
}
