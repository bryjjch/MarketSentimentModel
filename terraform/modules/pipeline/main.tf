# The daily ingestion pipeline:
#
#   EventBridge cron -> dispatch -> [collect queue] -> collect -> [predict queue]
#     -> predict --(high confidence)--> [cache-write queue] -> cache_write -> DynamoDB
#              \--(low confidence)---> [label queue] -> label -> s3://data/pseudo/
#
# Each stage lives in its own file; this one holds what they share.

locals {
  # The three actions a Lambda event source mapping needs on its source queue.
  sqs_consume_actions = [
    "sqs:ReceiveMessage",
    "sqs:DeleteMessage",
    "sqs:GetQueueAttributes",
  ]

  llm_secret_arns = compact([var.openai_secret_arn, var.google_secret_arn])

  # Every stage that writes to S3 needs to list the bucket first.
  list_data_bucket_statement = {
    Sid      = "ListDataBucket"
    Effect   = "Allow"
    Action   = ["s3:ListBucket"]
    Resource = var.data_bucket_arn
  }
}
