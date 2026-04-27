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

resource "aws_s3_object" "phrasebank" {
  bucket                 = aws_s3_bucket.data.id
  key                    = "reference/phrasebank/${basename(var.phrasebank_path)}"
  source                 = abspath(var.phrasebank_path)
  etag                   = filemd5(abspath(var.phrasebank_path))
  server_side_encryption = "AES256"
}
