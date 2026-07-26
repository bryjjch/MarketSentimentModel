# Queues between pipeline stages. Every stage is idempotent (same run_id => same S3
# keys; cache writes are conditional on updated_at), so redelivery is safe. Each queue
# has a DLQ so poison messages surface instead of retrying forever.
#
# Visibility timeouts are 6x the consumer Lambda timeout (AWS guidance for Lambda
# event source mappings).

# ---------------------------------------------------------------------------
# collect: pipeline_dispatch -> pipeline_collect
# ---------------------------------------------------------------------------

resource "aws_sqs_queue" "collect_dlq" {
  name                      = "${var.project_name}-collect-dlq"
  message_retention_seconds = 1209600 # 14 days
}

resource "aws_sqs_queue" "collect" {
  name                       = "${var.project_name}-collect"
  visibility_timeout_seconds = 720
  message_retention_seconds  = 345600 # 4 days

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.collect_dlq.arn
    maxReceiveCount     = 3
  })
}

# ---------------------------------------------------------------------------
# predict: pipeline_collect -> pipeline_predict
# ---------------------------------------------------------------------------

resource "aws_sqs_queue" "predict_dlq" {
  name                      = "${var.project_name}-predict-dlq"
  message_retention_seconds = 1209600
}

resource "aws_sqs_queue" "predict" {
  name                       = "${var.project_name}-predict"
  visibility_timeout_seconds = 1800
  message_retention_seconds  = 345600

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.predict_dlq.arn
    maxReceiveCount     = 3
  })
}

# ---------------------------------------------------------------------------
# label: pipeline_predict -> pipeline_label
# ---------------------------------------------------------------------------

resource "aws_sqs_queue" "label_dlq" {
  name                      = "${var.project_name}-label-dlq"
  message_retention_seconds = 1209600
}

resource "aws_sqs_queue" "label" {
  name                       = "${var.project_name}-label"
  visibility_timeout_seconds = 3600
  message_retention_seconds  = 345600

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.label_dlq.arn
    maxReceiveCount     = 3
  })
}

# ---------------------------------------------------------------------------
# cache-write: pipeline_predict + api_sentiment -> cache_write
# ---------------------------------------------------------------------------

resource "aws_sqs_queue" "cache_write_dlq" {
  name                      = "${var.project_name}-cache-write-dlq"
  message_retention_seconds = 1209600
}

resource "aws_sqs_queue" "cache_write" {
  name                       = "${var.project_name}-cache-write"
  visibility_timeout_seconds = 180
  message_retention_seconds  = 86400 # 1 day; stale sentiment is worthless

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.cache_write_dlq.arn
    maxReceiveCount     = 5
  })
}
