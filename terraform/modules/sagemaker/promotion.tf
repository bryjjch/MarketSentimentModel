# ---------------------------------------------------------------------------
# model_promote: repoints the endpoint when a model package is approved
#
# Owns the SageMaker model / endpoint config / endpoint that Terraform deliberately
# does not manage (see registry.tf).
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
