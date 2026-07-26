variable "project_name" {
  type        = string
  description = "Resource name prefix."
}

variable "aws_region" {
  type        = string
  description = "Region resources live in; used to build the SSM parameter ARN."
}

variable "aws_account_id" {
  type        = string
  description = "Account id; used to build the SSM parameter ARN."
}

variable "image_uris" {
  type = object({
    sentiment      = string
    cache_read     = string
    ticker_suggest = string
  })
  description = "Tagged ECR image URI per API Lambda."
}

# --- HTTP API ----------------------------------------------------------------

variable "cors_allow_origins" {
  type        = list(string)
  description = "Allowed origins for CORS (use specific origins in production)."
}

variable "throttle_rate_limit" {
  type        = number
  description = "Stage steady-state requests per second (default route throttling)."
}

variable "throttle_burst_limit" {
  type        = number
  description = "Stage short burst capacity (requests); typically >= rate limit."
}

# --- Wiring to other modules -------------------------------------------------

variable "sagemaker_endpoint_name" {
  type        = string
  description = "Endpoint the sentiment Lambda invokes."
}

variable "sagemaker_endpoint_arn" {
  type        = string
  description = "Endpoint ARN used to scope the InvokeEndpoint policy."
}

variable "sentiment_cache_table_name" {
  type        = string
  description = "DynamoDB sentiment cache table name."
}

variable "sentiment_cache_table_arn" {
  type        = string
  description = "DynamoDB sentiment cache table ARN."
}

variable "cache_write_queue_url" {
  type        = string
  description = "Queue the sentiment Lambda writes cache updates to."
}

variable "cache_write_queue_arn" {
  type        = string
  description = "Cache-write queue ARN used to scope the SendMessage policy."
}

# --- Behaviour ---------------------------------------------------------------

variable "reddit_credentials_secret_arn" {
  type        = string
  description = "Optional Secrets Manager ARN for Reddit API credentials. Empty disables Reddit access."
  default     = ""
}

variable "sentiment_cache_api_ttl_seconds" {
  type        = number
  description = "Seconds added to by-symbol write-back time for the DynamoDB expires_at attribute."
}

variable "rss_overfetch" {
  type        = number
  description = "Multiplier applied to max_articles when fetching Google News RSS."
}

variable "valid_tickers_ssm_param" {
  type        = string
  description = "Optional SSM parameter name holding the valid ticker universe. Empty falls back to valid_tickers_json / valid_tickers_file."
  default     = ""
}

variable "valid_tickers_json" {
  type        = string
  description = "Fallback JSON array of valid ticker symbols."
  default     = ""
}

variable "valid_tickers_file" {
  type        = string
  description = "Packaged ticker JSON file readable by Lambda."
}

variable "valid_tickers_cache_ttl_seconds" {
  type        = number
  description = "In-memory cache TTL (seconds) for the valid ticker universe inside Lambda execution environments."
}
