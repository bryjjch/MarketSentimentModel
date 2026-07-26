# ---------------------------------------------------------------------------
# cache_write: sole owner of DynamoDB sentiment_cache writes
#
# Fed by both the pipeline (predict) and the real-time API, which is why the queue is
# shared and this consumer reports partial batch failures instead of failing whole batches.
# ---------------------------------------------------------------------------

module "cache_write" {
  source = "../lambda-function"

  name        = "${var.project_name}-cache-write"
  image_uri   = var.image_uris.cache_write
  timeout     = 30
  memory_size = 128

  environment = {
    TABLE_NAME = var.sentiment_cache_table_name
  }

  policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "WriteSentimentCache"
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem"]
        Resource = var.sentiment_cache_table_arn
      },
      {
        Sid      = "ConsumeCacheWriteTasks"
        Effect   = "Allow"
        Action   = local.sqs_consume_actions
        Resource = var.queue_arns.cache_write
      },
    ]
  })
}

resource "aws_lambda_event_source_mapping" "cache_write" {
  event_source_arn        = var.queue_arns.cache_write
  function_name           = module.cache_write.arn
  batch_size              = 10
  function_response_types = ["ReportBatchItemFailures"]

  maximum_batching_window_in_seconds = 5
}
