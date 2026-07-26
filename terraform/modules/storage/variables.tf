variable "project_name" {
  type        = string
  description = "Resource name prefix."
}

variable "models_bucket_name" {
  type        = string
  description = "Bucket name for model artifacts."
}

variable "data_bucket_name" {
  type        = string
  description = "Bucket name for the raw/predictions/pseudo/curated pipeline data."
}

variable "force_destroy" {
  type        = bool
  description = "If true, empty and delete both buckets on terraform destroy (dev only)."
  default     = false
}

variable "data_retention_days" {
  type        = number
  description = "Lifecycle expiration (days) for the data bucket's raw/ and predictions/ prefixes. 0 disables the lifecycle rules."
  default     = 90
}
