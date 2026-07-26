# FinSense infrastructure. Each module owns one slice of the stack; this file is the
# only place they are wired together, so the dependency graph is readable top to bottom:
#
#   ecr ──────────────> (images for every Lambda)
#   storage ──────────> buckets + sentiment cache table
#   queues ───────────> stage-to-stage SQS
#   sagemaker ────────> SageMaker IAM, model registry, promotion, retrain schedule
#   pipeline ─────────> daily ingest -> collect -> predict -> label -> cache
#   api ──────────────> HTTP API and its Lambdas
#   github_oidc ──────> CI roles (bootstrap only; see the module for the import note)

module "ecr" {
  source = "./modules/ecr"

  project_name = var.project_name
  repositories = keys(local.lambda_images)
}

module "storage" {
  source = "./modules/storage"

  project_name        = var.project_name
  models_bucket_name  = local.bucket_name
  data_bucket_name    = local.data_bucket_name
  force_destroy       = var.s3_force_destroy
  data_retention_days = var.data_retention_days
}

module "queues" {
  source = "./modules/queues"

  project_name = var.project_name
}

module "sagemaker" {
  source = "./modules/sagemaker"

  project_name            = var.project_name
  aws_region              = var.aws_region
  aws_account_id          = local.account_id
  model_promote_image_uri = local.image_uris["model-promote"]

  models_bucket_arn = module.storage.models_bucket_arn
  data_bucket_arn   = module.storage.data_bucket_arn
  data_bucket       = module.storage.data_bucket

  endpoint_name                = local.endpoint_name
  endpoint_arn                 = local.endpoint_arn
  serverless_memory_size_in_mb = var.sagemaker_serverless_memory_size_in_mb
  serverless_max_concurrency   = var.sagemaker_serverless_max_concurrency
  model_versions_to_keep       = var.model_versions_to_keep

  model_package_group_name       = var.model_package_group_name
  pipeline_name                  = local.pipeline_name
  sagemaker_image_uri            = var.sagemaker_image_uri
  pipeline_macro_f1_threshold    = var.pipeline_macro_f1_threshold
  retrain_schedule               = var.retrain_schedule
  retrain_schedule_enabled       = var.retrain_schedule_enabled
  retrain_training_instance_type = var.retrain_training_instance_type
}

module "pipeline" {
  source = "./modules/pipeline"

  project_name = var.project_name

  image_uris = {
    dispatch    = local.image_uris["pipeline-dispatch"]
    collect     = local.image_uris["pipeline-collect"]
    predict     = local.image_uris["pipeline-predict"]
    label       = local.image_uris["pipeline-label"]
    cache_write = local.image_uris["cache-write"]
  }

  data_bucket                = module.storage.data_bucket
  data_bucket_arn            = module.storage.data_bucket_arn
  sentiment_cache_table_name = module.storage.sentiment_cache_table_name
  sentiment_cache_table_arn  = module.storage.sentiment_cache_table_arn

  queue_arns = {
    collect     = module.queues.queue_arns["collect"]
    predict     = module.queues.queue_arns["predict"]
    label       = module.queues.queue_arns["label"]
    cache_write = module.queues.queue_arns["cache_write"]
  }

  queue_urls = {
    collect     = module.queues.queue_urls["collect"]
    predict     = module.queues.queue_urls["predict"]
    label       = module.queues.queue_urls["label"]
    cache_write = module.queues.queue_urls["cache_write"]
  }

  sagemaker_endpoint_name = local.endpoint_name
  sagemaker_endpoint_arn  = local.endpoint_arn

  ingestion_schedule            = var.ingestion_schedule
  ingestion_max_articles        = var.ingestion_max_articles
  ingestion_include_social      = var.ingestion_include_social
  top_tickers_json              = var.top_tickers_json
  rss_overfetch                 = var.rss_overfetch
  reddit_credentials_secret_arn = var.reddit_credentials_secret_arn

  sagemaker_batch_size        = var.sagemaker_batch_size
  sentiment_cache_ttl_seconds = var.sentiment_cache_ttl_seconds
  low_conf_top_prob           = var.low_conf_top_prob
  low_conf_margin             = var.low_conf_margin

  llm_provider      = var.llm_provider
  llm_model         = var.llm_model
  llm_temperature   = var.llm_temperature
  llm_timeout_s     = var.llm_timeout_s
  llm_max_chars     = var.llm_max_chars
  llm_seed          = var.llm_seed
  openai_secret_arn = var.openai_secret_arn
  google_secret_arn = var.google_secret_arn

  collect_lambda_memory_mb = var.collect_lambda_memory_mb
  predict_lambda_memory_mb = var.predict_lambda_memory_mb
  label_lambda_memory_mb   = var.label_lambda_memory_mb
}

module "api" {
  source = "./modules/api"

  project_name   = var.project_name
  aws_region     = var.aws_region
  aws_account_id = local.account_id

  image_uris = {
    sentiment      = local.image_uris["api-sentiment"]
    cache_read     = local.image_uris["api-cache-read"]
    ticker_suggest = local.image_uris["api-ticker-suggest"]
  }

  cors_allow_origins   = var.cors_allow_origins
  throttle_rate_limit  = var.apigateway_throttle_rate_limit
  throttle_burst_limit = var.apigateway_throttle_burst_limit

  sagemaker_endpoint_name    = local.endpoint_name
  sagemaker_endpoint_arn     = local.endpoint_arn
  sentiment_cache_table_name = module.storage.sentiment_cache_table_name
  sentiment_cache_table_arn  = module.storage.sentiment_cache_table_arn
  cache_write_queue_url      = module.queues.queue_urls["cache_write"]
  cache_write_queue_arn      = module.queues.queue_arns["cache_write"]

  reddit_credentials_secret_arn   = var.reddit_credentials_secret_arn
  sentiment_cache_api_ttl_seconds = var.sentiment_cache_api_ttl_seconds
  rss_overfetch                   = var.rss_overfetch
  valid_tickers_ssm_param         = var.valid_tickers_ssm_param
  valid_tickers_json              = var.valid_tickers_json
  valid_tickers_file              = var.valid_tickers_file
  valid_tickers_cache_ttl_seconds = var.valid_tickers_cache_ttl_seconds
}

module "github_oidc" {
  source = "./modules/github-oidc"

  project_name          = var.project_name
  aws_region            = var.aws_region
  aws_account_id        = local.account_id
  github_repository     = var.github_repository
  github_default_branch = var.github_default_branch
  tf_state_bucket       = var.tf_state_bucket
  tf_lock_table         = var.tf_lock_table
}
