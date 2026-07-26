variable "project_name" {
  type        = string
  description = "Resource name prefix."
}

variable "image_uris" {
  type = object({
    dispatch    = string
    collect     = string
    predict     = string
    label       = string
    cache_write = string
  })
  description = "Tagged ECR image URI per pipeline Lambda."
}

# --- Wiring to other modules -------------------------------------------------

variable "data_bucket" {
  type        = string
  description = "Pipeline data bucket name."
}

variable "data_bucket_arn" {
  type        = string
  description = "Pipeline data bucket ARN."
}

variable "sentiment_cache_table_name" {
  type        = string
  description = "DynamoDB sentiment cache table name."
}

variable "sentiment_cache_table_arn" {
  type        = string
  description = "DynamoDB sentiment cache table ARN."
}

variable "queue_arns" {
  type = object({
    collect     = string
    predict     = string
    label       = string
    cache_write = string
  })
  description = "SQS queue ARNs between pipeline stages."
}

variable "queue_urls" {
  type = object({
    collect     = string
    predict     = string
    label       = string
    cache_write = string
  })
  description = "SQS queue URLs between pipeline stages."
}

variable "sagemaker_endpoint_name" {
  type        = string
  description = "Endpoint the predict Lambda invokes."
}

variable "sagemaker_endpoint_arn" {
  type        = string
  description = "Endpoint ARN used to scope the InvokeEndpoint policy."
}

# --- Ingestion ---------------------------------------------------------------

variable "ingestion_schedule" {
  type        = string
  description = "EventBridge rate/cron that triggers the dispatch Lambda."
}

variable "ingestion_max_articles" {
  type        = number
  description = "Max articles per ticker during daily ingestion."
}

variable "ingestion_include_social" {
  type        = bool
  description = "Whether daily ingestion should include Reddit (requires reddit_credentials_secret_arn)."
}

variable "top_tickers_json" {
  type        = string
  description = "JSON array of tickers stored in SSM for the dispatch Lambda."
}

variable "rss_overfetch" {
  type        = number
  description = "Multiplier applied to max_articles when fetching Google News RSS."
}

variable "reddit_credentials_secret_arn" {
  type        = string
  description = "Optional Secrets Manager ARN for Reddit API credentials. Empty disables Reddit access."
  default     = ""
}

# --- Prediction --------------------------------------------------------------

variable "sagemaker_batch_size" {
  type        = number
  description = "Batch size the predict Lambda uses per InvokeEndpoint call."
}

variable "sentiment_cache_ttl_seconds" {
  type        = number
  description = "Seconds added to refresh time for the DynamoDB expires_at attribute."
}

variable "low_conf_top_prob" {
  type        = number
  description = "Top-class probability below which a prediction is routed to pseudo-labeling."
}

variable "low_conf_margin" {
  type        = number
  description = "Minimum margin (top-prob - runner-up) required for a confident prediction; 0 disables the gate."
}

# --- Pseudo-labeling ---------------------------------------------------------

variable "llm_provider" {
  type        = string
  description = "Pseudo-labeling provider ('openai', 'google', or 'echo')."
}

variable "llm_model" {
  type        = string
  description = "Optional pseudo-labeling model id. Empty picks the provider default."
}

variable "llm_temperature" {
  type        = number
  description = "Sampling temperature for the pseudo-labeling LLM."
}

variable "llm_timeout_s" {
  type        = number
  description = "Per-request timeout (seconds) for the pseudo-labeling LLM."
}

variable "llm_max_chars" {
  type        = number
  description = "Maximum input characters sent to the pseudo-labeling LLM per row."
}

variable "llm_seed" {
  type        = number
  description = "Deterministic seed passed to the pseudo-labeling LLM where supported."
}

variable "openai_secret_arn" {
  type        = string
  description = "Optional Secrets Manager ARN for the OpenAI API key. Empty disables it."
  default     = ""
}

variable "google_secret_arn" {
  type        = string
  description = "Optional Secrets Manager ARN for the Google AI Studio API key. Empty disables it."
  default     = ""
}

# --- Sizing ------------------------------------------------------------------

variable "collect_lambda_memory_mb" {
  type        = number
  description = "Memory (MB) for the per-symbol collect Lambda."
}

variable "predict_lambda_memory_mb" {
  type        = number
  description = "Memory (MB) for the predict Lambda."
}

variable "label_lambda_memory_mb" {
  type        = number
  description = "Memory (MB) for the LLM label Lambda."
}
