# ---------------------------------------------------------------------------
# IAM: SageMaker inference execution role
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${var.project_name}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

data "aws_iam_policy_document" "sagemaker_s3_and_logs" {
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
      aws_s3_bucket.models.arn,
      "${aws_s3_bucket.models.arn}/*",
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
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
    resources = [
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*",
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*:log-stream:*",
    ]
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

resource "aws_iam_role_policy" "sagemaker_inline" {
  name   = "${var.project_name}-sagemaker-s3-logs-ecr"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_s3_and_logs.json
}

# ---------------------------------------------------------------------------
# IAM: SageMaker training pipeline execution role
# ---------------------------------------------------------------------------

resource "aws_iam_role" "sagemaker_pipeline" {
  name               = "${var.project_name}-sagemaker-pipeline"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

data "aws_iam_policy_document" "sagemaker_pipeline_permissions" {
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
      aws_s3_bucket.models.arn,
      "${aws_s3_bucket.models.arn}/*",
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
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
    resources = [
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*",
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*:log-stream:*",
    ]
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
      "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:training-job/*",
      "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:processing-job/*",
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
      "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:model-package/${var.model_package_group_name}/*",
      "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:model-package-group/${var.model_package_group_name}",
      "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:model/*",
    ]
  }

  statement {
    sid = "PassRoleToSagemaker"
    actions = [
      "iam:PassRole",
    ]
    resources = [
      aws_iam_role.sagemaker_pipeline.arn,
    ]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "sagemaker_pipeline_inline" {
  name   = "${var.project_name}-sagemaker-pipeline-perms"
  role   = aws_iam_role.sagemaker_pipeline.id
  policy = data.aws_iam_policy_document.sagemaker_pipeline_permissions.json
}

# ---------------------------------------------------------------------------
# SageMaker: inference model and endpoint
#
# The model, endpoint configuration and endpoint are NOT Terraform resources. They
# are created and rolled forward by the model_promote Lambda (see lambdas.tf) each
# time a model package is approved. Describing them here would mean:
#   - every deploy needed the 405 MB model.tar.gz on the machine running apply, and
#   - every retrain showed up as drift Terraform wanted to revert.
#
# Terraform still owns the endpoint's *name* (local.endpoint_name), its execution
# role, and its serverless sizing, which reach the promoter as env vars.
#
# Migrating an existing deployment: drop them from state without touching AWS —
#   terraform state rm aws_sagemaker_model.classifier \
#                      aws_sagemaker_endpoint_configuration.classifier \
#                      aws_sagemaker_endpoint.classifier \
#                      aws_s3_object.model_artifact
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SageMaker: training pipeline
# ---------------------------------------------------------------------------

resource "aws_sagemaker_model_package_group" "sentiment" {
  model_package_group_name = var.model_package_group_name
}

# The Pipeline itself is deliberately NOT a Terraform resource.
#
# Its definition is produced by the SageMaker SDK (src/sagemaker/pipeline/build_pipeline.py),
# which uploads sourcedir.tar.gz to S3 while compiling and bakes this role's ARN into
# the JSON. Managing it here meant Terraform needed a generated file that could only be
# generated after Terraform had run. CI breaks the cycle by applying Terraform first,
# then running `build_pipeline.py --upsert` with the role ARN from an output.
# See the sagemaker-pipeline job in .github/workflows/deploy.yml.
