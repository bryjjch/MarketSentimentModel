output "project_name" {
  description = "Resource name prefix. Read by scripts/lambda-build-push.sh to derive ECR repository names."
  value       = var.project_name
}

output "aws_region" {
  description = "Region every resource is deployed into."
  value       = var.aws_region
}

output "s3_bucket_name" {
  description = "Bucket storing model artifacts."
  value       = aws_s3_bucket.models.bucket
}

output "sagemaker_endpoint_name" {
  description = "Name passed to InvokeEndpoint / Lambda env SAGEMAKER_ENDPOINT_NAME"
  value       = local.endpoint_name
}

output "sagemaker_endpoint_arn" {
  value = local.endpoint_arn
}

output "http_api_invoke_url" {
  description = "Base URL for the HTTP API"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "sentiment_by_symbol_url" {
  description = "Full URL for POST /sentiment/by-symbol (JSON body: symbol, optional options)"
  value       = "${trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")}/sentiment/by-symbol"
}

output "sentiment_cache_read_url_template" {
  description = "GET cached snapshot: append symbol (e.g. .../sentiment/cache/AAPL)"
  value       = "${trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")}/sentiment/cache/{symbol}"
}

output "sentiment_cache_list_url" {
  description = "GET all active cached sentiment rows as a JSON array"
  value       = "${trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")}/sentiment/cache"
}

output "ticker_suggest_url_template" {
  description = "GET ticker suggestions by prefix (e.g. .../tickers/suggest?q=AAP&limit=10)"
  value       = "${trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")}/tickers/suggest?q={prefix}&limit={n}"
}

output "sentiment_cache_table_name" {
  description = "DynamoDB table for precomputed per-symbol sentiment"
  value       = aws_dynamodb_table.sentiment_cache.name
}

output "data_bucket_name" {
  description = "S3 bucket for raw/predictions/pseudo/curated pipeline data (separate from the model bucket)."
  value       = aws_s3_bucket.data.bucket
}

output "pipeline_dispatch_function_name" {
  description = "Dispatch Lambda (EventBridge cron target); enqueues one collect task per ticker."
  value       = aws_lambda_function.pipeline_dispatch.function_name
}

output "pipeline_queue_urls" {
  description = "SQS queue URLs between pipeline stages (collect -> predict -> label; cache-write is shared with the API)."
  value = {
    collect     = aws_sqs_queue.collect.url
    predict     = aws_sqs_queue.predict.url
    label       = aws_sqs_queue.label.url
    cache_write = aws_sqs_queue.cache_write.url
  }
}

output "pipeline_dispatch_rule_name" {
  description = "EventBridge rule name for the daily pipeline dispatch."
  value       = aws_cloudwatch_event_rule.pipeline_dispatch.name
}

output "pipeline_name" {
  description = "SageMaker Pipeline name. The pipeline is upserted by CI, not Terraform; this is the name CI upserts under."
  value       = local.pipeline_name
}

output "pipeline_role_arn" {
  description = "IAM role ARN used by the SageMaker Pipeline. Passed to build_pipeline.py --role."
  value       = aws_iam_role.sagemaker_pipeline.arn
}

output "model_package_group_name" {
  description = "SageMaker Model Package Group for registered model versions."
  value       = aws_sagemaker_model_package_group.sentiment.model_package_group_name
}
