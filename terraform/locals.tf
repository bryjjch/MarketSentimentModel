locals {
  account_id = data.aws_caller_identity.current.account_id

  # --- Names -----------------------------------------------------------------

  bucket_name = coalesce(
    var.bucket_name,
    "${var.project_name}-models-${local.account_id}"
  )
  data_bucket_name = coalesce(
    var.data_bucket_name,
    "${var.project_name}-data-${local.account_id}"
  )
  endpoint_name = coalesce(var.endpoint_name, "${var.project_name}-endpoint")
  pipeline_name = coalesce(var.pipeline_name, "${var.project_name}-training-pipeline")

  # The endpoint is created by model_promote, not Terraform, so its ARN is constructed
  # rather than referenced. Consumers only need it to scope InvokeEndpoint policies.
  endpoint_arn = "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:endpoint/${local.endpoint_name}"

  # --- Lambda images ---------------------------------------------------------

  # The single list of Lambda images: one ECR repository each, all sharing var.image_tag
  # (set by scripts/lambda-build-push.sh, or -var from CI). Adding a Lambda means adding
  # a key here and consuming local.image_uris[...] in the module that owns it.
  lambda_images = {
    "pipeline-dispatch"  = "cron entrypoint; fans tickers out onto the collect queue"
    "pipeline-collect"   = "per-symbol news/social collection into raw/"
    "pipeline-predict"   = "scores a raw partition, splits high/low confidence"
    "pipeline-label"     = "LLM pseudo-labeler for low-confidence rows"
    "cache-write"        = "sole writer of the DynamoDB sentiment cache"
    "api-sentiment"      = "POST /sentiment/by-symbol"
    "api-cache-read"     = "GET /sentiment/cache[/{symbol}]"
    "api-ticker-suggest" = "GET /tickers/suggest"
    "model-promote"      = "repoints the endpoint when a model package is approved"
  }

  image_uris = {
    for name, url in module.ecr.repository_urls : name => "${url}:${var.image_tag}"
  }
}
