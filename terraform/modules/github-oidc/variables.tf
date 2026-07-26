variable "project_name" {
  type        = string
  description = "Resource name prefix; also scopes what IAM the apply role may touch."
}

variable "aws_region" {
  type        = string
  description = "Region holding the state lock table."
}

variable "aws_account_id" {
  type        = string
  description = "Account id used to build role and lock table ARNs."
}

variable "github_repository" {
  type        = string
  description = "owner/repo allowed to assume the CI roles via OIDC."
}

variable "github_default_branch" {
  type        = string
  description = "Branch whose workflow runs may assume the apply role."
}

variable "tf_state_bucket" {
  type        = string
  description = "S3 bucket holding Terraform state (created out of band; see backend.hcl)."
}

variable "tf_lock_table" {
  type        = string
  description = "DynamoDB table used for Terraform state locking (created out of band; see backend.hcl)."
}
