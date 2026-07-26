# Output names are a public interface: scripts/ and .github/workflows/ read them by
# name (`terraform output -raw ...`). Rename one only alongside its consumers.

locals {
  api_base_url = trimsuffix(module.api.invoke_url, "/")
}

# --- Deployment identity -----------------------------------------------------

output "project_name" {
  description = "Resource name prefix. Read by scripts/lambda-build-push.sh to derive ECR repository names."
  value       = var.project_name
}

output "aws_region" {
  description = "Region every resource is deployed into."
  value       = var.aws_region
}

# --- Storage -----------------------------------------------------------------

output "s3_bucket_name" {
  description = "Bucket storing model artifacts."
  value       = module.storage.models_bucket
}

output "data_bucket_name" {
  description = "S3 bucket for raw/predictions/pseudo/curated pipeline data (separate from the model bucket)."
  value       = module.storage.data_bucket
}

output "sentiment_cache_table_name" {
  description = "DynamoDB table for precomputed per-symbol sentiment"
  value       = module.storage.sentiment_cache_table_name
}

# --- Inference ---------------------------------------------------------------

output "sagemaker_endpoint_name" {
  description = "Name passed to InvokeEndpoint / Lambda env SAGEMAKER_ENDPOINT_NAME"
  value       = local.endpoint_name
}

output "sagemaker_endpoint_arn" {
  value = local.endpoint_arn
}

# --- HTTP API ----------------------------------------------------------------

output "http_api_invoke_url" {
  description = "Base URL for the HTTP API"
  value       = module.api.invoke_url
}

output "sentiment_by_symbol_url" {
  description = "Full URL for POST /sentiment/by-symbol (JSON body: symbol, optional options)"
  value       = "${local.api_base_url}/sentiment/by-symbol"
}

output "sentiment_cache_read_url_template" {
  description = "GET cached snapshot: append symbol (e.g. .../sentiment/cache/AAPL)"
  value       = "${local.api_base_url}/sentiment/cache/{symbol}"
}

output "sentiment_cache_list_url" {
  description = "GET all active cached sentiment rows as a JSON array"
  value       = "${local.api_base_url}/sentiment/cache"
}

output "ticker_suggest_url_template" {
  description = "GET ticker suggestions by prefix (e.g. .../tickers/suggest?q=AAP&limit=10)"
  value       = "${local.api_base_url}/tickers/suggest?q={prefix}&limit={n}"
}

# --- Ingestion pipeline ------------------------------------------------------

output "pipeline_dispatch_function_name" {
  description = "Dispatch Lambda (EventBridge cron target); enqueues one collect task per ticker."
  value       = module.pipeline.dispatch_function_name
}

output "pipeline_dispatch_rule_name" {
  description = "EventBridge rule name for the daily pipeline dispatch."
  value       = module.pipeline.dispatch_rule_name
}

output "pipeline_queue_urls" {
  description = "SQS queue URLs between pipeline stages (collect -> predict -> label; cache-write is shared with the API)."
  value       = module.queues.queue_urls
}

# --- Training pipeline -------------------------------------------------------

output "pipeline_name" {
  description = "SageMaker Pipeline name. The pipeline is upserted by CI, not Terraform; this is the name CI upserts under."
  value       = local.pipeline_name
}

output "pipeline_role_arn" {
  description = "IAM role ARN used by the SageMaker Pipeline. Passed to build_pipeline.py --role."
  value       = module.sagemaker.pipeline_role_arn
}

output "model_package_group_name" {
  description = "SageMaker Model Package Group for registered model versions."
  value       = module.sagemaker.model_package_group_name
}

# --- CI ----------------------------------------------------------------------

output "github_plan_role_arn" {
  description = "Set as the GitHub Actions repository variable AWS_PLAN_ROLE_ARN."
  value       = module.github_oidc.plan_role_arn
}

output "github_apply_role_arn" {
  description = "Set as the GitHub Actions repository variable AWS_APPLY_ROLE_ARN."
  value       = module.github_oidc.apply_role_arn
}
