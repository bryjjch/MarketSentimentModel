# Root input variables.
#
# Terraform requires every root variable to be declared here, so this file covers all
# domains; the banners below mirror the modules in main.tf that consume them.

# ===========================================================================
# Core: deployment identity and storage
# ===========================================================================

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

# --- Storage -----------------------------------------------------------------

variable "bucket_name" {
  type        = string
  description = "Globally unique S3 bucket name for model artifacts. If empty, uses project_name-models-account_id."
  default     = null
}

variable "data_bucket_name" {
  type        = string
  description = "Globally unique S3 bucket name for the data pipeline (raw/predictions/pseudo/curated). If null, uses project_name-data-account_id."
  default     = null
}

variable "s3_force_destroy" {
  type        = bool
  description = "If true, empty and delete the model bucket on terraform destroy (dev only)."
  default     = false
}

variable "data_retention_days" {
  type        = number
  description = "Lifecycle expiration (days) for raw/ and predictions/ partitions. Pseudo/curated are retained indefinitely. Set to 0 to disable."
  default     = 90
}

# ===========================================================================
# Ingestion pipeline (modules/pipeline)
# ===========================================================================

# --- Ingestion ---------------------------------------------------------------

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

variable "top_tickers_json" {
  type        = string
  description = "JSON array of tickers stored in SSM at /{project_name}/top-tickers for the ingestion Lambda."
  default     = "[\"AAPL\",\"MSFT\",\"GOOGL\",\"META\",\"NVDA\"]"
}

# Shared with the API's sentiment Lambda, which fetches the same feeds on demand.
variable "rss_overfetch" {
  type        = number
  description = "Multiplier applied to max_articles when fetching Google News RSS, so post-filtering can drop noise without starving the result set. Hard-capped to 60 inside the Lambda."
  default     = 3
}

variable "reddit_credentials_secret_arn" {
  type        = string
  description = "Optional Secrets Manager ARN; secret must be JSON {\"client_id\":\"...\",\"client_secret\":\"...\"} for Reddit API (include_social). Leave empty to disable Reddit."
  default     = ""
}

# --- Prediction --------------------------------------------------------------

variable "sagemaker_batch_size" {
  type        = number
  description = "Batch size the prediction Lambda uses per InvokeEndpoint call."
  default     = 32
}

variable "sentiment_cache_ttl_seconds" {
  type        = number
  description = "Unix seconds added to refresh time for DynamoDB expires_at (TTL cleanup)."
  default     = 604800
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

# --- Pseudo-labeling ---------------------------------------------------------

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
  description = "Optional pseudo-labeling model id (e.g. gpt-4o-mini, gemini-3.1-flash-lite). Empty picks the provider default."
  default     = ""
}

variable "llm_temperature" {
  type        = number
  description = "Sampling temperature for the pseudo-labeling LLM."
  default     = 0.0
}

variable "llm_timeout_s" {
  type        = number
  description = "Per-request timeout (seconds) for the pseudo-labeling LLM."
  default     = 15
}

variable "llm_max_chars" {
  type        = number
  description = "Maximum input characters sent to the pseudo-labeling LLM per row."
  default     = 4000
}

variable "llm_seed" {
  type        = number
  description = "Deterministic seed passed to the pseudo-labeling LLM where supported."
  default     = 42
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

# --- Sizing ------------------------------------------------------------------

variable "collect_lambda_memory_mb" {
  type        = number
  description = "Memory (MB) for the per-symbol collect Lambda."
  default     = 512
}

variable "predict_lambda_memory_mb" {
  type        = number
  description = "Memory (MB) for the predict Lambda."
  default     = 512
}

variable "label_lambda_memory_mb" {
  type        = number
  description = "Memory (MB) for the LLM label Lambda."
  default     = 512
}

# ===========================================================================
# HTTP API (modules/api)
# ===========================================================================

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

variable "sentiment_cache_api_ttl_seconds" {
  type        = number
  description = "Unix seconds added to API by-symbol write-back time for DynamoDB expires_at."
  default     = 86400
}

# --- Ticker universe ---------------------------------------------------------
# Resolution order inside the Lambdas: SSM parameter, then inline JSON, then the
# file packaged in the image.

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

variable "valid_tickers_file" {
  type        = string
  description = "Path to a packaged ticker JSON file readable by Lambda. Relative paths resolve against finsense_shared/tickers/data/ inside the image."
  default     = "valid_tickers_us.json"
}

variable "valid_tickers_cache_ttl_seconds" {
  type        = number
  description = "In-memory cache TTL (seconds) for the valid ticker universe inside Lambda execution environments."
  default     = 900
}

# ===========================================================================
# SageMaker: endpoint, registry, training pipeline (modules/sagemaker)
# ===========================================================================

# --- Inference endpoint ------------------------------------------------------

variable "sagemaker_image_uri" {
  type        = string
  description = "Hugging Face PyTorch inference DLC image URI for this region. Passed to the training pipeline as its InferenceImageUri parameter; the promoted model package carries it onto the endpoint."
}

variable "endpoint_name" {
  type        = string
  description = "SageMaker endpoint name (must be unique in the account/region). If null, derived from project_name."
  default     = null
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

# --- Model registry ----------------------------------------------------------

variable "model_package_group_name" {
  type        = string
  description = "SageMaker Model Package Group name for registered model versions."
  default     = "finsense-sentiment"
}

variable "model_versions_to_keep" {
  type        = number
  description = "How many SageMaker model / endpoint-config generations model_promote leaves in place. Older ones are pruned; keeping a few makes rollback a single re-approval."
  default     = 3

  validation {
    condition     = var.model_versions_to_keep >= 1
    error_message = "model_versions_to_keep must be at least 1 so the live endpoint config is never pruned."
  }
}

# --- Training pipeline -------------------------------------------------------

variable "pipeline_name" {
  type        = string
  description = "SageMaker Pipeline name. If null, derived from project_name."
  default     = null
}

variable "pipeline_macro_f1_threshold" {
  type        = number
  description = "Minimum macro F1 required by the pipeline ConditionStep to register a model. Passed to scheduled runs as the MacroF1Threshold pipeline parameter."
  default     = 0.80
}

variable "retrain_schedule" {
  type        = string
  description = "EventBridge schedule expression for automatic retraining runs."
  default     = "cron(0 6 1 * ? *)" # 06:00 UTC on the 1st of each month
}

variable "retrain_schedule_enabled" {
  type        = bool
  description = "Whether the scheduled retraining rule is enabled. Defaults to false so the first runs are started deliberately."
  default     = false
}

variable "retrain_training_instance_type" {
  type        = string
  description = "Instance type for the scheduled pipeline's training steps."
  default     = "ml.g4dn.xlarge"
}

# ===========================================================================
# GitHub Actions OIDC (modules/github-oidc)
# ===========================================================================

variable "github_repository" {
  type        = string
  description = "owner/repo allowed to assume the CI roles via OIDC."
  default     = "bryjjch/FinSense"

  validation {
    condition     = can(regex("^[^/]+/[^/]+$", var.github_repository))
    error_message = "github_repository must be in owner/repo form."
  }
}

variable "github_default_branch" {
  type        = string
  description = "Branch whose workflow runs may assume the apply role."
  default     = "main"
}

variable "tf_state_bucket" {
  type        = string
  description = "S3 bucket holding Terraform state (created out of band; see backend.hcl)."
  default     = "finsense-terraform-state-bucket"
}

variable "tf_lock_table" {
  type        = string
  description = "DynamoDB table used for Terraform state locking (created out of band; see backend.hcl)."
  default     = "finsense-terraform-states-table"
}
