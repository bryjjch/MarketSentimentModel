data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

locals {
  sagemaker_log_group_arns = [
    "arn:aws:logs:${var.aws_region}:${var.aws_account_id}:log-group:/aws/sagemaker/*",
    "arn:aws:logs:${var.aws_region}:${var.aws_account_id}:log-group:/aws/sagemaker/*:log-stream:*",
  ]

  sagemaker_arn_prefix = "arn:aws:sagemaker:${var.aws_region}:${var.aws_account_id}"
}

# ---------------------------------------------------------------------------
# Inference execution role (assumed by the endpoint)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "execution" {
  name               = "${var.project_name}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

data "aws_iam_policy_document" "execution" {
  # Both buckets: pipeline training jobs write model.tar.gz under the data bucket's
  # pipeline prefix, and that is the artifact a promoted model package points at.
  # Models-bucket-only access here fails the endpoint at model-download time.
  statement {
    sid = "ModelArtifactRead"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = [
      var.models_bucket_arn,
      "${var.models_bucket_arn}/*",
      var.data_bucket_arn,
      "${var.data_bucket_arn}/*",
    ]
  }

  statement {
    sid = "CloudWatchLogs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = local.sagemaker_log_group_arns
  }

  statement {
    sid = "ECRReadForDLC"
    actions = [
      "ecr:GetAuthorizationToken",
    ]
    resources = ["*"]
  }

  statement {
    sid = "ECRBatchGetImage"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "execution" {
  name   = "${var.project_name}-sagemaker-s3-logs-ecr"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution.json
}

# ---------------------------------------------------------------------------
# Training pipeline execution role (assumed by pipeline steps)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "pipeline" {
  name               = "${var.project_name}-sagemaker-pipeline"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

data "aws_iam_policy_document" "pipeline" {
  statement {
    sid = "S3ArtifactReadWrite"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = [
      var.models_bucket_arn,
      "${var.models_bucket_arn}/*",
      var.data_bucket_arn,
      "${var.data_bucket_arn}/*",
    ]
  }

  statement {
    sid = "CloudWatchLogs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = local.sagemaker_log_group_arns
  }

  statement {
    sid = "ECRAccess"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = ["*"]
  }

  statement {
    sid = "SageMakerTrainingAndProcessing"
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:StopProcessingJob",
      "sagemaker:AddTags",
    ]
    resources = [
      "${local.sagemaker_arn_prefix}:training-job/*",
      "${local.sagemaker_arn_prefix}:processing-job/*",
    ]
  }

  statement {
    sid = "ModelPackageRegistration"
    actions = [
      "sagemaker:CreateModelPackage",
      "sagemaker:DescribeModelPackage",
      "sagemaker:UpdateModelPackage",
      "sagemaker:DescribeModelPackageGroup",
      "sagemaker:CreateModel",
      "sagemaker:DescribeModel",
    ]
    resources = [
      "${local.sagemaker_arn_prefix}:model-package/${var.model_package_group_name}/*",
      "${local.sagemaker_arn_prefix}:model-package-group/${var.model_package_group_name}",
      "${local.sagemaker_arn_prefix}:model/*",
    ]
  }

  statement {
    sid = "PassRoleToSagemaker"
    actions = [
      "iam:PassRole",
    ]
    resources = [
      aws_iam_role.pipeline.arn,
    ]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "pipeline" {
  name   = "${var.project_name}-sagemaker-pipeline-perms"
  role   = aws_iam_role.pipeline.id
  policy = data.aws_iam_policy_document.pipeline.json
}
