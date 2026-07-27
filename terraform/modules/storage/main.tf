# Everything in the project that holds state:
#
#   s3://<models bucket>  -> model artifacts, written by SageMaker training jobs
#   s3://<data bucket>    -> the pipeline's raw/predictions/pseudo/curated partitions
#   DynamoDB              -> the sentiment cache the API reads
#
# The two buckets share a baseline (private, encrypted, versioned, TLS-only) and are
# deliberately declared separately rather than through a shared bucket module. They are
# the only two buckets the project has and they already diverge — only the data bucket
# carries lifecycle rules — so declaring each in full reads end to end without a hop
# through a wrapper. The cost is that a change to the baseline has to be made twice.

# ---------------------------------------------------------------------------
# S3: model artifacts
#
# The model artifact is not uploaded from a workstation. SageMaker training jobs write
# it, the pipeline registers it as a model package, and model_promote points the
# endpoint at the approved version. See modules/sagemaker.
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "models" {
  bucket        = var.models_bucket_name
  force_destroy = var.force_destroy
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

data "aws_iam_policy_document" "models_deny_insecure_transport" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.models.arn, "${aws_s3_bucket.models.arn}/*"]
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "models_tls" {
  bucket = aws_s3_bucket.models.id
  policy = data.aws_iam_policy_document.models_deny_insecure_transport.json

  # The public access block must land first, or the policy write is rejected.
  depends_on = [aws_s3_bucket_public_access_block.models]
}

# ---------------------------------------------------------------------------
# S3: pipeline data (raw / predictions / pseudo / curated)
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "data" {
  bucket        = var.data_bucket_name
  force_destroy = var.force_destroy
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

data "aws_iam_policy_document" "data_deny_insecure_transport" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "data_tls" {
  bucket = aws_s3_bucket.data.id
  policy = data.aws_iam_policy_document.data_deny_insecure_transport.json

  # The public access block must land first, or the policy write is rejected.
  depends_on = [aws_s3_bucket_public_access_block.data]
}

# Expire raw + predictions after a configurable number of days to control cost. Pseudo
# and curated partitions are retained indefinitely because they double as training data.
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  count  = var.data_retention_days > 0 ? 1 : 0
  bucket = aws_s3_bucket.data.id

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
  depends_on = [aws_s3_bucket_versioning.data]
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
