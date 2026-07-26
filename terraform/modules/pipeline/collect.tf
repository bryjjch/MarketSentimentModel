# ---------------------------------------------------------------------------
# collect: per-symbol news/social collection into raw/
# ---------------------------------------------------------------------------

module "collect" {
  source = "../lambda-function"

  name        = "${var.project_name}-pipeline-collect"
  image_uri   = var.image_uris.collect
  timeout     = 120
  memory_size = var.collect_lambda_memory_mb

  environment = {
    DATA_BUCKET          = var.data_bucket
    PREDICT_QUEUE_URL    = var.queue_urls.predict
    DEFAULT_MAX_ARTICLES = tostring(var.ingestion_max_articles)
    INCLUDE_SOCIAL       = var.ingestion_include_social ? "true" : "false"
    REDDIT_SECRET_ARN    = var.reddit_credentials_secret_arn
    RSS_OVERFETCH        = tostring(var.rss_overfetch)
  }

  policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid    = "WriteRawData"
          Effect = "Allow"
          Action = [
            "s3:PutObject",
            "s3:AbortMultipartUpload",
          ]
          Resource = "${var.data_bucket_arn}/raw/*"
        },
        local.list_data_bucket_statement,
        {
          Sid      = "ConsumeCollectTasks"
          Effect   = "Allow"
          Action   = local.sqs_consume_actions
          Resource = var.queue_arns.collect
        },
        {
          Sid      = "SendPredictTasks"
          Effect   = "Allow"
          Action   = ["sqs:SendMessage"]
          Resource = var.queue_arns.predict
        },
      ],
      var.reddit_credentials_secret_arn != "" ? [
        {
          Sid      = "ReadRedditSecret"
          Effect   = "Allow"
          Action   = ["secretsmanager:GetSecretValue"]
          Resource = var.reddit_credentials_secret_arn
        }
      ] : []
    )
  })
}

resource "aws_lambda_event_source_mapping" "collect" {
  event_source_arn = var.queue_arns.collect
  function_name    = module.collect.arn
  batch_size       = 1

  scaling_config {
    # Keep the fleet polite towards Google News RSS / Reddit rate limits.
    maximum_concurrency = 3
  }
}
