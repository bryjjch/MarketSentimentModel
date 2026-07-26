# ---------------------------------------------------------------------------
# Scheduled retraining: starts the SageMaker Pipeline directly, no CI involved
#
# Disabled by default (var.retrain_schedule_enabled). Turn it on once you have
# watched a manually started run finish and approved its model package.
# ---------------------------------------------------------------------------

locals {
  # The pipeline is upserted by CI rather than managed here, so its ARN is constructed.
  pipeline_arn = "${local.sagemaker_arn_prefix}:pipeline/${var.pipeline_name}"
}

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
