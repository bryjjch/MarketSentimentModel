terraform {
  required_version = ">= 1.5"

  # Configure remote state in S3 via -backend-config=backend.hcl during init.
  backend "s3" {}

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}
