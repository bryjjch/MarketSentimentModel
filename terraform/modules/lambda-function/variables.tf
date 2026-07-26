variable "name" {
  type        = string
  description = "Fully qualified function name (already prefixed with project_name). The IAM role and inline policy names are derived from it."
}

variable "image_uri" {
  type        = string
  description = "ECR image URI including tag."
}

variable "timeout" {
  type        = number
  description = "Function timeout in seconds."
}

variable "memory_size" {
  type        = number
  description = "Function memory in MB."
}

variable "environment" {
  type        = map(string)
  description = "Environment variables. An empty map omits the environment block entirely."
  default     = {}
}

variable "policy_json" {
  type        = string
  description = "Inline IAM policy JSON for the execution role. Null creates no inline policy (logs-only function)."
  default     = null
}
