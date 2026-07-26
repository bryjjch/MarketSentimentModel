output "models_bucket" {
  description = "Model artifact bucket name."
  value       = module.models.bucket
}

output "models_bucket_arn" {
  description = "Model artifact bucket ARN."
  value       = module.models.arn
}

output "data_bucket" {
  description = "Pipeline data bucket name."
  value       = module.data.bucket
}

output "data_bucket_arn" {
  description = "Pipeline data bucket ARN."
  value       = module.data.arn
}

output "sentiment_cache_table_name" {
  description = "DynamoDB table holding precomputed per-symbol sentiment."
  value       = aws_dynamodb_table.sentiment_cache.name
}

output "sentiment_cache_table_arn" {
  description = "DynamoDB sentiment cache table ARN."
  value       = aws_dynamodb_table.sentiment_cache.arn
}
