variable "name" {
  type        = string
  description = "Globally unique bucket name."
}

variable "force_destroy" {
  type        = bool
  description = "If true, empty and delete the bucket on terraform destroy (dev only)."
  default     = false
}
