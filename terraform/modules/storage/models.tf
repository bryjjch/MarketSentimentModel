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
