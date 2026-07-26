variable "image_tag" {
  type        = string
  description = "ECR image tag for all Lambda container images. Set automatically by lambda-build-push.sh via image_tag.auto.tfvars."
}

variable "aws_region" {
  type        = string
  description = "AWS region for all resources (must match SageMaker DLC URI region)."
}

variable "project_name" {
  type        = string
  description = "Short name prefix for resource names (letters, numbers, hyphen)."
  default     = "finsense"
}

variable "bucket_name" {
  type        = string
  description = "Globally unique S3 bucket name for model artifacts. If empty, uses project_name-models-account_id."
  default     = null
}

variable "model_key_prefix" {
  type        = string
  description = "S3 key prefix (no leading slash) for the packaged model object."
  default     = "models/finsense/v1"
}

variable "model_tarball_path" {
  type        = string
  description = "Local path to model.tar.gz produced by a SageMaker training job (auto-packaged from SM_MODEL_DIR). May be relative to the directory where you run terraform apply."
}

variable "sagemaker_image_uri" {
  type        = string
  description = "Hugging Face PyTorch inference DLC image URI for this region (see infra/README.md)."
}

variable "sagemaker_serverless_memory_size_in_mb" {
  type        = number
  description = "Memory size for SageMaker Serverless Inference (MB). Valid values are 1024, 2048, 3072, 4096, 5120, or 6144."
  default     = 2048
}

variable "sagemaker_serverless_max_concurrency" {
  type        = number
  description = "Maximum concurrent invocations for SageMaker Serverless Inference."
  default     = 10
}

variable "endpoint_name" {
  type        = string
  description = "SageMaker endpoint name (must be unique in the account/region). If null, derived from project_name."
  default     = null
}

variable "cors_allow_origins" {
  type        = list(string)
  description = "Allowed origins for API Gateway HTTP API CORS (use specific origins in production)."
  default     = ["*"]
}

variable "apigateway_throttle_rate_limit" {
  type        = number
  description = "HTTP API stage steady-state requests per second (default route throttling)."
  default     = 10
}

variable "apigateway_throttle_burst_limit" {
  type        = number
  description = "HTTP API stage short burst capacity (requests); typically >= rate_limit."
  default     = 20
}

variable "lambda_reserved_concurrent_executions" {
  type        = number
  description = "Reserved concurrent executions for the predict Lambda (caps invocations in parallel). Use -1 for no reservation. Align with sagemaker_serverless_max_concurrency."
  default     = 5
}

variable "s3_force_destroy" {
  type        = bool
  description = "If true, empty and delete the model bucket on terraform destroy (dev only)."
  default     = false
}

variable "reddit_credentials_secret_arn" {
  type        = string
  description = "Optional Secrets Manager ARN; secret must be JSON {\"client_id\":\"...\",\"client_secret\":\"...\"} for Reddit API (include_social). Leave empty to disable Reddit."
  default     = ""
}

variable "sentiment_cache_ttl_seconds" {
  type        = number
  description = "Unix seconds added to refresh time for DynamoDB expires_at (TTL cleanup)."
  default     = 604800
}

variable "rss_overfetch" {
  type        = number
  description = "Multiplier applied to max_articles when fetching Google News RSS, so post-filtering can drop noise without starving the result set. Hard-capped to 60 inside the Lambda."
  default     = 3
}

variable "sentiment_cache_api_ttl_seconds" {
  type        = number
  description = "Unix seconds added to API by-symbol write-back time for DynamoDB expires_at."
  default     = 86400
}

variable "top_tickers_json" {
  type        = string
  description = "JSON array of tickers stored in SSM at /{project_name}/top-tickers for the ingestion Lambda."
  default     = "[\"AAPL\",\"MSFT\",\"GOOGL\",\"META\",\"NVDA\"]"
}

variable "valid_tickers_ssm_param" {
  type        = string
  description = "Optional SSM parameter name containing a JSON array of valid ticker symbols used by API validation/suggestions."
  default     = ""
}

variable "valid_tickers_json" {
  type        = string
  description = "Fallback JSON array of valid ticker symbols for API validation/suggestions when SSM is not used."
  default     = ""
}

variable "valid_tickers_cache_ttl_seconds" {
  type        = number
  description = "In-memory cache TTL (seconds) for the valid ticker universe inside Lambda execution environments."
  default     = 900
}

variable "valid_tickers_file" {
  type        = string
  description = "Path to a packaged ticker JSON file readable by Lambda. Relative paths resolve against the finsense_shared package directory inside the image."
  default     = "valid_tickers_us.json"
}

# --- Data bucket + daily ingestion / pseudo-labeling pipeline ---------------

variable "data_bucket_name" {
  type        = string
  description = "Globally unique S3 bucket name for the data pipeline (raw/predictions/pseudo/curated). If null, uses project_name-data-account_id."
  default     = null
}

variable "data_retention_days" {
  type        = number
  description = "Lifecycle expiration (days) for raw/ and predictions/ partitions. Pseudo/curated are retained indefinitely. Set to 0 to disable."
  default     = 90
}

variable "ingestion_schedule" {
  type        = string
  description = "EventBridge rate/cron that triggers the daily ingestion Lambda (e.g. 'cron(0 13 * * ? *)' for 13:00 UTC)."
  default     = "cron(0 13 * * ? *)"
}

variable "ingestion_max_articles" {
  type        = number
  description = "Max articles per ticker during daily ingestion."
  default     = 20
}

variable "ingestion_include_social" {
  type        = bool
  description = "Whether daily ingestion should include Reddit (requires reddit_credentials_secret_arn)."
  default     = true
}

variable "sagemaker_batch_size" {
  type        = number
  description = "Batch size the prediction Lambda uses per InvokeEndpoint call."
  default     = 32
}

variable "low_conf_top_prob" {
  type        = number
  description = "Top-class probability threshold below which a prediction is routed to pseudo-labeling (0.0 to 1.0)."
  default     = 0.65
}

variable "low_conf_margin" {
  type        = number
  description = "Minimum margin (top-prob - runner-up) required for a confident prediction; 0 disables the margin gate."
  default     = 0.0
}

variable "llm_provider" {
  type        = string
  description = "Pseudo-labeling provider ('openai', 'google', or 'echo')."
  default     = "openai"
  validation {
    condition     = contains(["openai", "google", "echo"], var.llm_provider)
    error_message = "llm_provider must be one of: openai, google, echo."
  }
}

variable "llm_model" {
  type        = string
  description = "Optional pseudo-labeling model id (e.g. gpt-4o-mini, gemini-2.0-flash). Empty picks the provider default."
  default     = ""
}

variable "openai_secret_arn" {
  type        = string
  description = "Optional Secrets Manager ARN for OpenAI API key (JSON with {\"api_key\":\"...\"}). Leave empty to disable."
  default     = ""
}

variable "google_secret_arn" {
  type        = string
  description = "Optional Secrets Manager ARN for Google AI Studio API key (JSON with {\"api_key\":\"...\"}). Leave empty to disable."
  default     = ""
}

variable "ingestion_lambda_memory_mb" {
  type        = number
  description = "Memory (MB) for the daily ingestion Lambda."
  default     = 512
}

variable "prediction_lambda_memory_mb" {
  type        = number
  description = "Memory (MB) for the prediction Lambda."
  default     = 512
}

variable "pseudo_label_lambda_memory_mb" {
  type        = number
  description = "Memory (MB) for the pseudo-label Lambda."
  default     = 512
}

variable "phrasebank_path" {
  type        = string
  description = "Local path to the Financial PhraseBank Sentences_75Agree.txt file, uploaded to the data bucket under reference/phrasebank/."
  default     = "../data/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
}

# --- SageMaker training pipeline --------------------------------------------

variable "pipeline_name" {
  type        = string
  description = "SageMaker Pipeline name. If null, derived from project_name."
  default     = null
}

variable "pipeline_definition_json" {
  type        = string
  description = "Inline pipeline definition JSON (takes precedence over pipeline_definition_path when non-empty)."
  default     = ""
}

variable "pipeline_definition_path" {
  type        = string
  description = "Path to the pipeline definition JSON file generated by build_pipeline.py. Used when pipeline_definition_json is empty."
  default     = "pipeline_definition.json"
}

variable "model_package_group_name" {
  type        = string
  description = "SageMaker Model Package Group name for registered model versions."
  default     = "finsense-sentiment"
}

variable "pipeline_macro_f1_threshold" {
  type        = number
  description = "Minimum macro F1 required by the pipeline ConditionStep to register a model."
  default     = 0.80
}
