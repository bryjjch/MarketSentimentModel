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
  statement {
    sid = "ModelArtifactRead"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.models.arn,
      "${aws_s3_bucket.models.arn}/*",
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
# ---------------------------------------------------------------------------

resource "aws_sagemaker_model" "classifier" {
  name               = "${var.project_name}-hf-classifier"
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    image          = var.sagemaker_image_uri
    model_data_url = "s3://${aws_s3_bucket.models.bucket}/${aws_s3_object.model_artifact.key}"
  }

  depends_on = [
    aws_s3_object.model_artifact,
    aws_iam_role_policy.sagemaker_inline,
  ]
}

resource "aws_sagemaker_endpoint_configuration" "classifier" {
  name = "${var.project_name}-ep-cfg"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.classifier.name
    initial_variant_weight = 1

    serverless_config {
      memory_size_in_mb = var.sagemaker_serverless_memory_size_in_mb
      max_concurrency   = var.sagemaker_serverless_max_concurrency
    }
  }
}

resource "aws_sagemaker_endpoint" "classifier" {
  name                 = local.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.classifier.name

  depends_on = [
    aws_sagemaker_endpoint_configuration.classifier,
  ]
}

# ---------------------------------------------------------------------------
# SageMaker: training pipeline
# ---------------------------------------------------------------------------

resource "aws_sagemaker_model_package_group" "sentiment" {
  model_package_group_name = var.model_package_group_name
}

resource "aws_sagemaker_pipeline" "training" {
  pipeline_name         = local.pipeline_name
  pipeline_display_name = "${var.project_name}-sentiment-training-pipeline"
  role_arn              = aws_iam_role.sagemaker_pipeline.arn

  pipeline_definition = var.pipeline_definition_json != "" ? var.pipeline_definition_json : file(var.pipeline_definition_path)

  depends_on = [
    aws_iam_role_policy.sagemaker_pipeline_inline,
    aws_sagemaker_model_package_group.sentiment,
  ]
}
