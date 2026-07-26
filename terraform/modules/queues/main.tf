# Queues between pipeline stages. Every stage is idempotent (same run_id => same S3
# keys; cache writes are conditional on updated_at), so redelivery is safe. Each queue
# has a DLQ so poison messages surface instead of retrying forever.
#
# Visibility timeouts are 6x the consumer Lambda timeout (AWS guidance for Lambda event
# source mappings), which is why they are declared here next to the queue rather than
# left to a default.

resource "aws_sqs_queue" "dlq" {
  for_each = var.queues

  name                      = "${var.project_name}-${each.value.name}-dlq"
  message_retention_seconds = each.value.dlq_retention_seconds
}

resource "aws_sqs_queue" "this" {
  for_each = var.queues

  name                       = "${var.project_name}-${each.value.name}"
  visibility_timeout_seconds = each.value.visibility_timeout_seconds
  message_retention_seconds  = each.value.message_retention_seconds

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq[each.key].arn
    maxReceiveCount     = each.value.max_receive_count
  })
}
