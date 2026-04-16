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
  description = "Local path to model.tar.gz (run ../scripts/package_model_tarball.py first). May be relative to the directory where you run terraform apply."
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
