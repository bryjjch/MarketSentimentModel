# The SageMaker side of the stack: the two IAM roles, the model package group, the
# model_promote Lambda that rolls the endpoint forward, and the retrain schedule.

locals {
  sagemaker_log_group_arns = [
    "arn:aws:logs:${var.aws_region}:${var.aws_account_id}:log-group:/aws/sagemaker/*",
    "arn:aws:logs:${var.aws_region}:${var.aws_account_id}:log-group:/aws/sagemaker/*:log-stream:*",
  ]

  sagemaker_arn_prefix = "arn:aws:sagemaker:${var.aws_region}:${var.aws_account_id}"

  # The pipeline is upserted by CI rather than managed here, so its ARN is constructed.
  pipeline_arn = "${local.sagemaker_arn_prefix}:pipeline/${var.pipeline_name}"
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
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

# ---------------------------------------------------------------------------
# Model registry
#
# The model, endpoint configuration and endpoint are NOT Terraform resources. They
# are created and rolled forward by the model_promote Lambda below each time a model
# package is approved. Describing them here would mean:
#   - every deploy needed the 405 MB model.tar.gz on the machine running apply, and
#   - every retrain showed up as drift Terraform wanted to revert.
#
# Terraform still owns the endpoint's *name*, its execution role, and its serverless
# sizing, which reach the promoter as env vars.
#
# Migrating an existing deployment: drop them from state without touching AWS —
#   terraform state rm aws_sagemaker_model.classifier \
#                      aws_sagemaker_endpoint_configuration.classifier \
#                      aws_sagemaker_endpoint.classifier \
#                      aws_s3_object.model_artifact
#
# The Pipeline itself is likewise not a Terraform resource. Its definition is produced
# by the SageMaker SDK (src/sagemaker/pipeline/build_pipeline.py), which uploads
# sourcedir.tar.gz to S3 while compiling and bakes the pipeline role's ARN into the
# JSON. Managing it here meant Terraform needed a generated file that could only be
# generated after Terraform had run. CI breaks the cycle by applying Terraform first,
# then running `build_pipeline.py --upsert` with the role ARN from an output.
# See the sagemaker-pipeline job in .github/workflows/deploy.yml.
# ---------------------------------------------------------------------------

resource "aws_sagemaker_model_package_group" "this" {
  model_package_group_name = var.model_package_group_name
}

# ---------------------------------------------------------------------------
# model_promote: repoints the endpoint when a model package is approved
#
# Owns the SageMaker model / endpoint config / endpoint that Terraform deliberately
# does not manage (see the model registry note above).
# ---------------------------------------------------------------------------

module "model_promote" {
  source = "../lambda-function"

  name      = "${var.project_name}-model-promote"
  image_uri = var.model_promote_image_uri
  # CreateEndpoint/UpdateEndpoint return as soon as the update is accepted; the
  # rollout itself happens asynchronously, so this needs no long timeout.
  timeout     = 60
  memory_size = 256

  environment = {
    PROJECT_NAME               = var.project_name
    ENDPOINT_NAME              = var.endpoint_name
    MODEL_PACKAGE_GROUP        = var.model_package_group_name
    EXECUTION_ROLE_ARN         = aws_iam_role.execution.arn
    SERVERLESS_MEMORY_MB       = tostring(var.serverless_memory_size_in_mb)
    SERVERLESS_MAX_CONCURRENCY = tostring(var.serverless_max_concurrency)
    KEEP_VERSIONS              = tostring(var.model_versions_to_keep)
  }

  policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadModelPackages"
        Effect = "Allow"
        Action = [
          "sagemaker:DescribeModelPackage",
          "sagemaker:ListModelPackages",
        ]
        Resource = [
          "${local.sagemaker_arn_prefix}:model-package/${var.model_package_group_name}/*",
          aws_sagemaker_model_package_group.this.arn,
        ]
      },
      {
        Sid    = "ManageModels"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateModel",
          "sagemaker:DescribeModel",
          "sagemaker:DeleteModel",
        ]
        Resource = "${local.sagemaker_arn_prefix}:model/${var.project_name}-model-v*"
      },
      {
        Sid    = "ManageEndpointConfigs"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DeleteEndpointConfig",
        ]
        Resource = "${local.sagemaker_arn_prefix}:endpoint-config/${var.project_name}-ep-cfg-v*"
      },
      {
        Sid    = "ManageEndpoint"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateEndpoint",
          "sagemaker:UpdateEndpoint",
          "sagemaker:DescribeEndpoint",
        ]
        Resource = var.endpoint_arn
      },
      {
        # list_models / list_endpoint_configs are account-scoped calls; the handler
        # filters by name prefix and the Delete* statements above are what actually
        # bound what it can remove.
        Sid    = "ListForPruning"
        Effect = "Allow"
        Action = [
          "sagemaker:ListModels",
          "sagemaker:ListEndpointConfigs",
        ]
        Resource = "*"
      },
      {
        Sid      = "PassExecutionRole"
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = aws_iam_role.execution.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      },
    ]
  })
}

# --- Trigger: a package in the FinSense group reaching Approved ---------------

resource "aws_cloudwatch_event_rule" "model_approved" {
  name        = "${var.project_name}-model-approved"
  description = "Fires model_promote when a package in the FinSense group is approved."

  event_pattern = jsonencode({
    source        = ["aws.sagemaker"]
    "detail-type" = ["SageMaker Model Package State Change"]
    detail = {
      ModelPackageGroupName = [var.model_package_group_name]
      ModelApprovalStatus   = ["Approved"]
    }
  })
}

resource "aws_cloudwatch_event_target" "model_approved" {
  rule      = aws_cloudwatch_event_rule.model_approved.name
  target_id = "ModelPromoteLambda"
  arn       = module.model_promote.arn
}

resource "aws_lambda_permission" "model_promote_events" {
  statement_id  = "AllowEventBridgeInvokeModelPromote"
  action        = "lambda:InvokeFunction"
  function_name = module.model_promote.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.model_approved.arn
}

# ---------------------------------------------------------------------------
# Scheduled retraining: starts the SageMaker Pipeline directly, no CI involved
#
# Disabled by default (var.retrain_schedule_enabled). Turn it on once you have
# watched a manually started run finish and approved its model package.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "events_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "retrain_invoke" {
  name               = "${var.project_name}-retrain-invoke"
  description        = "Lets EventBridge start executions of the training pipeline."
  assume_role_policy = data.aws_iam_policy_document.events_assume.json
}

resource "aws_iam_role_policy" "retrain_invoke" {
  name = "${var.project_name}-retrain-invoke-perms"
  role = aws_iam_role.retrain_invoke.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "StartTrainingPipeline"
        Effect   = "Allow"
        Action   = ["sagemaker:StartPipelineExecution"]
        Resource = local.pipeline_arn
      },
    ]
  })
}

resource "aws_cloudwatch_event_rule" "retrain" {
  name                = "${var.project_name}-retrain"
  description         = "Scheduled retraining run of the sentiment training pipeline."
  schedule_expression = var.retrain_schedule
  state               = var.retrain_schedule_enabled ? "ENABLED" : "DISABLED"
}

resource "aws_cloudwatch_event_target" "retrain" {
  rule      = aws_cloudwatch_event_rule.retrain.name
  target_id = "TrainingPipeline"
  arn       = local.pipeline_arn
  role_arn  = aws_iam_role.retrain_invoke.arn

  sagemaker_pipeline_target {
    # Only the parameters Terraform is the source of truth for; everything else
    # falls back to the defaults compiled into the pipeline definition.
    pipeline_parameter_list {
      name  = "DataBucket"
      value = var.data_bucket
    }
    pipeline_parameter_list {
      name  = "ModelPackageGroup"
      value = var.model_package_group_name
    }
    pipeline_parameter_list {
      name  = "MacroF1Threshold"
      value = tostring(var.pipeline_macro_f1_threshold)
    }
    pipeline_parameter_list {
      name  = "InferenceImageUri"
      value = var.sagemaker_image_uri
    }
    pipeline_parameter_list {
      name  = "TrainingInstanceType"
      value = var.retrain_training_instance_type
    }
  }
}
