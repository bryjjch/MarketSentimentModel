# ---------------------------------------------------------------------------
# label: LLM-backed labeler for low-confidence predictions
# ---------------------------------------------------------------------------

module "label" {
  source = "../lambda-function"

  name        = "${var.project_name}-pipeline-label"
  image_uri   = var.image_uris.label
  timeout     = 600
  memory_size = var.label_lambda_memory_mb

  environment = {
    DATA_BUCKET       = var.data_bucket
    LLM_PROVIDER      = var.llm_provider
    LLM_MODEL         = var.llm_model
    OPENAI_SECRET_ARN = var.openai_secret_arn
    GOOGLE_SECRET_ARN = var.google_secret_arn
    LLM_TEMPERATURE   = tostring(var.llm_temperature)
    LLM_TIMEOUT_S     = tostring(var.llm_timeout_s)
    LLM_MAX_CHARS     = tostring(var.llm_max_chars)
    LLM_SEED          = tostring(var.llm_seed)
  }

  policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid      = "ReadPredictions"
          Effect   = "Allow"
          Action   = ["s3:GetObject"]
          Resource = "${var.data_bucket_arn}/predictions/*"
        },
        {
          Sid    = "WritePseudoAndCurated"
          Effect = "Allow"
          Action = [
            "s3:PutObject",
            "s3:AbortMultipartUpload",
          ]
          Resource = [
            "${var.data_bucket_arn}/pseudo/*",
            "${var.data_bucket_arn}/curated/*",
          ]
        },
        local.list_data_bucket_statement,
        {
          Sid      = "ConsumeLabelTasks"
          Effect   = "Allow"
          Action   = local.sqs_consume_actions
          Resource = var.queue_arns.label
        },
      ],
      length(local.llm_secret_arns) > 0 ? [
        {
          Sid      = "ReadLLMSecrets"
          Effect   = "Allow"
          Action   = ["secretsmanager:GetSecretValue"]
          Resource = local.llm_secret_arns
        }
      ] : []
    )
  })
}

resource "aws_lambda_event_source_mapping" "label" {
  event_source_arn = var.queue_arns.label
  function_name    = module.label.arn
  batch_size       = 1

  scaling_config {
    # LLM providers rate-limit aggressively; two workers is plenty for a daily run.
    maximum_concurrency = 2
  }
}
