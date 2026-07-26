variable "project_name" {
  type        = string
  description = "Resource name prefix."
}

variable "aws_region" {
  type        = string
  description = "Region every resource is deployed into."
}

variable "aws_account_id" {
  type        = string
  description = "Account id, used to build SageMaker and CloudWatch ARNs."
}

variable "model_promote_image_uri" {
  type        = string
  description = "Tagged ECR image URI for the model_promote Lambda."
}

# --- Wiring to other modules -------------------------------------------------

variable "models_bucket_arn" {
  type        = string
  description = "Model artifact bucket ARN."
}

variable "data_bucket_arn" {
  type        = string
  description = "Pipeline data bucket ARN; training jobs write model.tar.gz beneath it."
}

variable "data_bucket" {
  type        = string
  description = "Pipeline data bucket name, passed to scheduled pipeline runs as DataBucket."
}

# --- Endpoint ----------------------------------------------------------------

variable "endpoint_name" {
  type        = string
  description = "Endpoint name model_promote creates and rolls forward."
}

variable "endpoint_arn" {
  type        = string
  description = "Endpoint ARN, constructed by the caller since the endpoint is not a Terraform resource."
}

variable "serverless_memory_size_in_mb" {
  type        = number
  description = "Memory size for SageMaker Serverless Inference (MB)."
}

variable "serverless_max_concurrency" {
  type        = number
  description = "Maximum concurrent invocations for SageMaker Serverless Inference."
}

variable "model_versions_to_keep" {
  type        = number
  description = "How many model / endpoint-config generations model_promote leaves in place."
}

# --- Registry and training pipeline ------------------------------------------

variable "model_package_group_name" {
  type        = string
  description = "Model Package Group name for registered model versions."
}

variable "pipeline_name" {
  type        = string
  description = "SageMaker Pipeline name. The pipeline is upserted by CI, so this is only used to build its ARN."
}

variable "sagemaker_image_uri" {
  type        = string
  description = "Hugging Face PyTorch inference DLC URI passed to scheduled runs as InferenceImageUri."
}

variable "pipeline_macro_f1_threshold" {
  type        = number
  description = "Minimum macro F1 required by the pipeline ConditionStep to register a model."
}

variable "retrain_schedule" {
  type        = string
  description = "EventBridge schedule expression for automatic retraining runs."
}

variable "retrain_schedule_enabled" {
  type        = bool
  description = "Whether the scheduled retraining rule is enabled."
}

variable "retrain_training_instance_type" {
  type        = string
  description = "Instance type for the scheduled pipeline's training steps."
}
