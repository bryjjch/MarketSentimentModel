# The daily ingestion pipeline:
#
#   EventBridge cron -> dispatch -> [collect queue] -> collect -> [predict queue]
#     -> predict --(high confidence)--> [cache-write queue] -> cache_write -> DynamoDB
#              \--(low confidence)---> [label queue] -> label -> s3://data/pseudo/
#
# The stages below are declared in that order; the locals hold what they share.

locals {
  # The three actions a Lambda event source mapping needs on its source queue.
  sqs_consume_actions = [
    "sqs:ReceiveMessage",
    "sqs:DeleteMessage",
    "sqs:GetQueueAttributes",
  ]

  llm_secret_arns = compact([var.openai_secret_arn, var.google_secret_arn])

  # Every stage that writes to S3 needs to list the bucket first.
  list_data_bucket_statement = {
    Sid      = "ListDataBucket"
    Effect   = "Allow"
    Action   = ["s3:ListBucket"]
    Resource = var.data_bucket_arn
  }
}

# ---------------------------------------------------------------------------
# dispatch: cron entrypoint; enumerates tickers and enqueues one collect task each
# ---------------------------------------------------------------------------

resource "aws_ssm_parameter" "top_tickers" {
  name  = "/${var.project_name}/top-tickers"
  type  = "String"
  value = var.top_tickers_json

  description = "JSON array of tickers for sentiment cache refresher (edit in console or terraform)."
}

module "dispatch" {
  source = "../lambda-function"

  name        = "${var.project_name}-pipeline-dispatch"
  image_uri   = var.image_uris.dispatch
  timeout     = 60
  memory_size = 128

  environment = {
    COLLECT_QUEUE_URL     = var.queue_urls.collect
    DEFAULT_MAX_ARTICLES  = tostring(var.ingestion_max_articles)
    INCLUDE_SOCIAL        = var.ingestion_include_social ? "true" : "false"
    TOP_TICKERS_SSM_PARAM = aws_ssm_parameter.top_tickers.name
    DEFAULT_TICKERS_JSON  = var.top_tickers_json
  }

  policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadTickerParam"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = aws_ssm_parameter.top_tickers.arn
      },
      {
        Sid      = "SendCollectTasks"
        Effect   = "Allow"
        Action   = ["sqs:SendMessage"]
        Resource = var.queue_arns.collect
      },
    ]
  })
}

# --- Daily trigger -----------------------------------------------------------

resource "aws_cloudwatch_event_rule" "dispatch" {
  name                = "${var.project_name}-pipeline-dispatch"
  description         = "Daily trigger for the dispatch Lambda, which enqueues one collect task per ticker."
  schedule_expression = var.ingestion_schedule
}

resource "aws_cloudwatch_event_target" "dispatch" {
  rule      = aws_cloudwatch_event_rule.dispatch.name
  target_id = "PipelineDispatchLambda"
  arn       = module.dispatch.arn
}

resource "aws_lambda_permission" "dispatch_events" {
  statement_id  = "AllowEventBridgeInvokePipelineDispatch"
  action        = "lambda:InvokeFunction"
  function_name = module.dispatch.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.dispatch.arn
}

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

# ---------------------------------------------------------------------------
# cache_write: sole owner of DynamoDB sentiment_cache writes
#
# Fed by both the pipeline (predict) and the real-time API, which is why the queue is
# shared and this consumer reports partial batch failures instead of failing whole batches.
# ---------------------------------------------------------------------------

module "cache_write" {
  source = "../lambda-function"

  name        = "${var.project_name}-cache-write"
  image_uri   = var.image_uris.cache_write
  timeout     = 30
  memory_size = 128

  environment = {
    TABLE_NAME = var.sentiment_cache_table_name
  }

  policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "WriteSentimentCache"
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem"]
        Resource = var.sentiment_cache_table_arn
      },
      {
        Sid      = "ConsumeCacheWriteTasks"
        Effect   = "Allow"
        Action   = local.sqs_consume_actions
        Resource = var.queue_arns.cache_write
      },
    ]
  })
}

resource "aws_lambda_event_source_mapping" "cache_write" {
  event_source_arn        = var.queue_arns.cache_write
  function_name           = module.cache_write.arn
  batch_size              = 10
  function_response_types = ["ReportBatchItemFailures"]

  maximum_batching_window_in_seconds = 5
}
