# ---------------------------------------------------------------------------
# predict: score one raw partition, split high/low confidence
# ---------------------------------------------------------------------------

module "predict" {
  source = "../lambda-function"

  name        = "${var.project_name}-pipeline-predict"
  image_uri   = var.image_uris.predict
  timeout     = 300
  memory_size = var.predict_lambda_memory_mb

  environment = {
    SAGEMAKER_ENDPOINT_NAME = var.sagemaker_endpoint_name
    DATA_BUCKET             = var.data_bucket
    LABEL_QUEUE_URL         = var.queue_urls.label
    CACHE_WRITE_QUEUE_URL   = var.queue_urls.cache_write
    CACHE_TTL_SECONDS       = tostring(var.sentiment_cache_ttl_seconds)
    RECENT_HEADLINES_MAX    = "10"
    LOW_CONF_TOP_PROB       = tostring(var.low_conf_top_prob)
    LOW_CONF_MARGIN         = tostring(var.low_conf_margin)
    SAGEMAKER_BATCH_SIZE    = tostring(var.sagemaker_batch_size)
  }

  policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadRawData"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${var.data_bucket_arn}/raw/*"
      },
      {
        Sid    = "WritePredictionsAndCurated"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:AbortMultipartUpload",
        ]
        Resource = [
          "${var.data_bucket_arn}/predictions/*",
          "${var.data_bucket_arn}/curated/*",
        ]
      },
      local.list_data_bucket_statement,
      {
        Sid      = "InvokeSageMakerEndpoint"
        Effect   = "Allow"
        Action   = ["sagemaker:InvokeEndpoint"]
        Resource = var.sagemaker_endpoint_arn
      },
      {
        Sid      = "ConsumePredictTasks"
        Effect   = "Allow"
        Action   = local.sqs_consume_actions
        Resource = var.queue_arns.predict
      },
      {
        Sid    = "SendLabelAndCacheWriteTasks"
        Effect = "Allow"
        Action = ["sqs:SendMessage"]
        Resource = [
          var.queue_arns.label,
          var.queue_arns.cache_write,
        ]
      },
    ]
  })
}

resource "aws_lambda_event_source_mapping" "predict" {
  event_source_arn = var.queue_arns.predict
  function_name    = module.predict.arn
  batch_size       = 1

  scaling_config {
    # Stay below sagemaker_serverless_max_concurrency so the API keeps headroom.
    maximum_concurrency = 5
  }
}
