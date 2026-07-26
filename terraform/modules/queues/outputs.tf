output "queue_arns" {
  description = "Logical key -> queue ARN."
  value       = { for key, queue in aws_sqs_queue.this : key => queue.arn }
}

output "queue_urls" {
  description = "Logical key -> queue URL."
  value       = { for key, queue in aws_sqs_queue.this : key => queue.url }
}

output "dlq_arns" {
  description = "Logical key -> dead-letter queue ARN."
  value       = { for key, queue in aws_sqs_queue.dlq : key => queue.arn }
}
