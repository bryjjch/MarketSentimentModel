# ---------------------------------------------------------------------------
# S3: model artifacts
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "models" {
  bucket        = local.bucket_name
  force_destroy = var.s3_force_destroy
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

  depends_on = [aws_s3_bucket_public_access_block.models]
}

# The model artifact is no longer uploaded from a workstation. SageMaker training jobs
# write it, the pipeline registers it as a model package, and model_promote points the
# endpoint at the approved version. See the note at the top of sagemaker.tf.

# ---------------------------------------------------------------------------
# S3: pipeline data (raw / predictions / pseudo / curated)
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "data" {
  bucket        = local.data_bucket_name
  force_destroy = var.s3_force_destroy
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

  depends_on = [aws_s3_bucket_public_access_block.data]
}

# Expire raw + predictions after a configurable number of days to control cost. Pseudo
# and curated partitions are retained indefinitely because they double as training data.
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  count  = var.data_retention_days > 0 ? 1 : 0
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "expire-raw"
    status = "Enabled"
    filter {
      prefix = "raw/"
    }
    expiration {
      days = var.data_retention_days
    }
    noncurrent_version_expiration {
      noncurrent_days = var.data_retention_days
    }
  }

  rule {
    id     = "expire-predictions"
    status = "Enabled"
    filter {
      prefix = "predictions/"
    }
    expiration {
      days = var.data_retention_days
    }
    noncurrent_version_expiration {
      noncurrent_days = var.data_retention_days
    }
  }

  depends_on = [aws_s3_bucket_versioning.data]
}

# The Financial PhraseBank corpus lives at s3://<data bucket>/reference/phrasebank/ and is
# seeded once by hand (see README). Terraform used to upload it from a local path, but
# `filemd5()` reads that file on every plan, so CI — which has no copy of a licence-gated
# 466 KB corpus — could not plan at all. The pipeline reads the whole prefix as a
# ProcessingInput (PhraseBankS3Prefix), so it never needed a Terraform-managed object.
#
# `destroy = false` makes Terraform forget the object instead of deleting it from S3.
# Safe to delete this block once it has been applied on main.
removed {
  from = aws_s3_object.phrasebank

  lifecycle {
    destroy = false
  }
}

# ---------------------------------------------------------------------------
# DynamoDB: sentiment cache
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

resource "aws_ssm_parameter" "top_tickers" {
  name  = "/${var.project_name}/top-tickers"
  type  = "String"
  value = var.top_tickers_json

  description = "JSON array of tickers for sentiment cache refresher (edit in console or terraform)."
}
