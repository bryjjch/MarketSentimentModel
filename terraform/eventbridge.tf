resource "aws_cloudwatch_event_rule" "pipeline_dispatch" {
  name                = "${var.project_name}-pipeline-dispatch"
  description         = "Daily trigger for the dispatch Lambda, which enqueues one collect task per ticker."
  schedule_expression = var.ingestion_schedule
}

resource "aws_cloudwatch_event_target" "pipeline_dispatch" {
  rule      = aws_cloudwatch_event_rule.pipeline_dispatch.name
  target_id = "PipelineDispatchLambda"
  arn       = aws_lambda_function.pipeline_dispatch.arn
}

resource "aws_lambda_permission" "eventbridge_invoke_pipeline_dispatch" {
  statement_id  = "AllowEventBridgeInvokePipelineDispatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pipeline_dispatch.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.pipeline_dispatch.arn
}

# ---------------------------------------------------------------------------
# Model promotion: approving a model package repoints the endpoint
# ---------------------------------------------------------------------------

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
  arn       = aws_lambda_function.model_promote.arn
}

resource "aws_lambda_permission" "eventbridge_invoke_model_promote" {
  statement_id  = "AllowEventBridgeInvokeModelPromote"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.model_promote.function_name
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
        Resource = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:pipeline/${local.pipeline_name}"
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
  # The pipeline is upserted by CI rather than managed here, so the ARN is constructed.
  arn      = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:pipeline/${local.pipeline_name}"
  role_arn = aws_iam_role.retrain_invoke.arn

  sagemaker_pipeline_target {
    # Only the parameters Terraform is the source of truth for; everything else
    # falls back to the defaults compiled into the pipeline definition.
    pipeline_parameter_list {
      name  = "DataBucket"
      value = aws_s3_bucket.data.bucket
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
