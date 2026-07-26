# ---------------------------------------------------------------------------
# S3: model artifacts
#
# The model artifact is not uploaded from a workstation. SageMaker training jobs write
# it, the pipeline registers it as a model package, and model_promote points the
# endpoint at the approved version. See modules/sagemaker.
# ---------------------------------------------------------------------------

module "models" {
  source = "../s3-bucket"

  name          = var.models_bucket_name
  force_destroy = var.force_destroy
}

# ---------------------------------------------------------------------------
# S3: pipeline data (raw / predictions / pseudo / curated)
# ---------------------------------------------------------------------------

module "data" {
  source = "../s3-bucket"

  name          = var.data_bucket_name
  force_destroy = var.force_destroy
}

# Expire raw + predictions after a configurable number of days to control cost. Pseudo
# and curated partitions are retained indefinitely because they double as training data.
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  count  = var.data_retention_days > 0 ? 1 : 0
  bucket = module.data.id

  # A list, not a set: `rule` is an ordered block, and reordering it would show up as
  # a diff against the deployed configuration.
  dynamic "rule" {
    for_each = ["raw", "predictions"]

    content {
      id     = "expire-${rule.value}"
      status = "Enabled"
      filter {
        prefix = "${rule.value}/"
      }
      expiration {
        days = var.data_retention_days
      }
      noncurrent_version_expiration {
        noncurrent_days = var.data_retention_days
      }
    }
  }

  # noncurrent_version_expiration is only meaningful once versioning is on.
  depends_on = [module.data]
}

# The Financial PhraseBank corpus lives at s3://<data bucket>/reference/phrasebank/ and is
# seeded once by hand (see README). Terraform used to upload it from a local path, but
# `filemd5()` reads that file on every plan, so CI — which has no copy of a licence-gated
# 466 KB corpus — could not plan at all. The pipeline reads the whole prefix as a
# ProcessingInput (PhraseBankS3Prefix), so it never needed a Terraform-managed object.

# ---------------------------------------------------------------------------
# DynamoDB: sentiment cache
#
# cache_write is the sole writer; the API reads it. TTL cleanup runs off expires_at.
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "sentiment_cache" {
  name         = "${var.project_name}-sentiment-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "symbol"

  attribute {
    name = "symbol"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}
