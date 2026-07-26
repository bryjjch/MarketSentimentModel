# Shared assume-role policy document used by all Lambda IAM roles.
data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

locals {
  llm_secret_arns = compact([var.openai_secret_arn, var.google_secret_arn])
  sqs_consume_actions = [
    "sqs:ReceiveMessage",
    "sqs:DeleteMessage",
    "sqs:GetQueueAttributes",
  ]
}

# ---------------------------------------------------------------------------
# pipeline_dispatch: cron entrypoint; enumerates tickers and enqueues collect tasks
# ---------------------------------------------------------------------------

resource "aws_iam_role" "pipeline_dispatch_lambda" {
  name               = "${var.project_name}-pipeline-dispatch-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "pipeline_dispatch_lambda_basic" {
  role       = aws_iam_role.pipeline_dispatch_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "pipeline_dispatch_lambda" {
  name = "${var.project_name}-pipeline-dispatch-perms"
  role = aws_iam_role.pipeline_dispatch_lambda.id

  policy = jsonencode({
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
        Resource = aws_sqs_queue.collect.arn
      },
    ]
  })
}

resource "aws_lambda_function" "pipeline_dispatch" {
  function_name = "${var.project_name}-pipeline-dispatch"
  role          = aws_iam_role.pipeline_dispatch_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.pipeline_dispatch.repository_url}:${var.image_tag}"

  timeout     = 60
  memory_size = 128

  environment {
    variables = {
      COLLECT_QUEUE_URL     = aws_sqs_queue.collect.url
      DEFAULT_MAX_ARTICLES  = tostring(var.ingestion_max_articles)
      INCLUDE_SOCIAL        = var.ingestion_include_social ? "true" : "false"
      TOP_TICKERS_SSM_PARAM = aws_ssm_parameter.top_tickers.name
      DEFAULT_TICKERS_JSON  = var.top_tickers_json
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.pipeline_dispatch_lambda_basic,
    aws_iam_role_policy.pipeline_dispatch_lambda,
    aws_ssm_parameter.top_tickers,
  ]
}

# ---------------------------------------------------------------------------
# pipeline_collect: per-symbol news/social collection into raw/
# ---------------------------------------------------------------------------

resource "aws_iam_role" "pipeline_collect_lambda" {
  name               = "${var.project_name}-pipeline-collect-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "pipeline_collect_lambda_basic" {
  role       = aws_iam_role.pipeline_collect_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "pipeline_collect_lambda" {
  name = "${var.project_name}-pipeline-collect-perms"
  role = aws_iam_role.pipeline_collect_lambda.id

  policy = jsonencode({
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
          Resource = "${aws_s3_bucket.data.arn}/raw/*"
        },
        {
          Sid      = "ListDataBucket"
          Effect   = "Allow"
          Action   = ["s3:ListBucket"]
          Resource = aws_s3_bucket.data.arn
        },
        {
          Sid      = "ConsumeCollectTasks"
          Effect   = "Allow"
          Action   = local.sqs_consume_actions
          Resource = aws_sqs_queue.collect.arn
        },
        {
          Sid      = "SendPredictTasks"
          Effect   = "Allow"
          Action   = ["sqs:SendMessage"]
          Resource = aws_sqs_queue.predict.arn
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

resource "aws_lambda_function" "pipeline_collect" {
  function_name = "${var.project_name}-pipeline-collect"
  role          = aws_iam_role.pipeline_collect_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.pipeline_collect.repository_url}:${var.image_tag}"

  timeout     = 120
  memory_size = var.collect_lambda_memory_mb

  environment {
    variables = {
      DATA_BUCKET          = aws_s3_bucket.data.bucket
      PREDICT_QUEUE_URL    = aws_sqs_queue.predict.url
      DEFAULT_MAX_ARTICLES = tostring(var.ingestion_max_articles)
      INCLUDE_SOCIAL       = var.ingestion_include_social ? "true" : "false"
      REDDIT_SECRET_ARN    = var.reddit_credentials_secret_arn
      RSS_OVERFETCH        = tostring(var.rss_overfetch)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.pipeline_collect_lambda_basic,
    aws_iam_role_policy.pipeline_collect_lambda,
    aws_s3_bucket.data,
  ]
}

resource "aws_lambda_event_source_mapping" "collect" {
  event_source_arn = aws_sqs_queue.collect.arn
  function_name    = aws_lambda_function.pipeline_collect.arn
  batch_size       = 1

  scaling_config {
    # Keep the fleet polite towards Google News RSS / Reddit rate limits.
    maximum_concurrency = 3
  }
}

# ---------------------------------------------------------------------------
# pipeline_predict: score one raw partition, split high/low confidence
# ---------------------------------------------------------------------------

resource "aws_iam_role" "pipeline_predict_lambda" {
  name               = "${var.project_name}-pipeline-predict-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "pipeline_predict_lambda_basic" {
  role       = aws_iam_role.pipeline_predict_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "pipeline_predict_lambda" {
  name = "${var.project_name}-pipeline-predict-perms"
  role = aws_iam_role.pipeline_predict_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadRawData"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.data.arn}/raw/*"
      },
      {
        Sid    = "WritePredictionsAndCurated"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:AbortMultipartUpload",
        ]
        Resource = [
          "${aws_s3_bucket.data.arn}/predictions/*",
          "${aws_s3_bucket.data.arn}/curated/*",
        ]
      },
      {
        Sid      = "ListDataBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.data.arn
      },
      {
        Sid      = "InvokeSageMakerEndpoint"
        Effect   = "Allow"
        Action   = ["sagemaker:InvokeEndpoint"]
        Resource = aws_sagemaker_endpoint.classifier.arn
      },
      {
        Sid      = "ConsumePredictTasks"
        Effect   = "Allow"
        Action   = local.sqs_consume_actions
        Resource = aws_sqs_queue.predict.arn
      },
      {
        Sid    = "SendLabelAndCacheWriteTasks"
        Effect = "Allow"
        Action = ["sqs:SendMessage"]
        Resource = [
          aws_sqs_queue.label.arn,
          aws_sqs_queue.cache_write.arn,
        ]
      },
    ]
  })
}

resource "aws_lambda_function" "pipeline_predict" {
  function_name = "${var.project_name}-pipeline-predict"
  role          = aws_iam_role.pipeline_predict_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.pipeline_predict.repository_url}:${var.image_tag}"

  timeout     = 300
  memory_size = var.predict_lambda_memory_mb

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = aws_sagemaker_endpoint.classifier.name
      DATA_BUCKET             = aws_s3_bucket.data.bucket
      LABEL_QUEUE_URL         = aws_sqs_queue.label.url
      CACHE_WRITE_QUEUE_URL   = aws_sqs_queue.cache_write.url
      CACHE_TTL_SECONDS       = tostring(var.sentiment_cache_ttl_seconds)
      RECENT_HEADLINES_MAX    = "10"
      LOW_CONF_TOP_PROB       = tostring(var.low_conf_top_prob)
      LOW_CONF_MARGIN         = tostring(var.low_conf_margin)
      SAGEMAKER_BATCH_SIZE    = tostring(var.sagemaker_batch_size)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.pipeline_predict_lambda_basic,
    aws_iam_role_policy.pipeline_predict_lambda,
    aws_s3_bucket.data,
    aws_sagemaker_endpoint.classifier,
  ]
}

resource "aws_lambda_event_source_mapping" "predict" {
  event_source_arn = aws_sqs_queue.predict.arn
  function_name    = aws_lambda_function.pipeline_predict.arn
  batch_size       = 1

  scaling_config {
    # Stay below sagemaker_serverless_max_concurrency so the API keeps headroom.
    maximum_concurrency = 5
  }
}

# ---------------------------------------------------------------------------
# pipeline_label: LLM-backed labeler for low-confidence predictions
# ---------------------------------------------------------------------------

resource "aws_iam_role" "pipeline_label_lambda" {
  name               = "${var.project_name}-pipeline-label-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "pipeline_label_lambda_basic" {
  role       = aws_iam_role.pipeline_label_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "pipeline_label_lambda" {
  name = "${var.project_name}-pipeline-label-perms"
  role = aws_iam_role.pipeline_label_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid      = "ReadPredictions"
          Effect   = "Allow"
          Action   = ["s3:GetObject"]
          Resource = "${aws_s3_bucket.data.arn}/predictions/*"
        },
        {
          Sid    = "WritePseudoAndCurated"
          Effect = "Allow"
          Action = [
            "s3:PutObject",
            "s3:AbortMultipartUpload",
          ]
          Resource = [
            "${aws_s3_bucket.data.arn}/pseudo/*",
            "${aws_s3_bucket.data.arn}/curated/*",
          ]
        },
        {
          Sid      = "ListDataBucket"
          Effect   = "Allow"
          Action   = ["s3:ListBucket"]
          Resource = aws_s3_bucket.data.arn
        },
        {
          Sid      = "ConsumeLabelTasks"
          Effect   = "Allow"
          Action   = local.sqs_consume_actions
          Resource = aws_sqs_queue.label.arn
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

resource "aws_lambda_function" "pipeline_label" {
  function_name = "${var.project_name}-pipeline-label"
  role          = aws_iam_role.pipeline_label_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.pipeline_label.repository_url}:${var.image_tag}"

  timeout     = 600
  memory_size = var.label_lambda_memory_mb

  environment {
    variables = {
      DATA_BUCKET       = aws_s3_bucket.data.bucket
      LLM_PROVIDER      = var.llm_provider
      LLM_MODEL         = var.llm_model
      OPENAI_SECRET_ARN = var.openai_secret_arn
      GOOGLE_SECRET_ARN = var.google_secret_arn
      LLM_TEMPERATURE   = tostring(var.llm_temperature)
      LLM_TIMEOUT_S     = tostring(var.llm_timeout_s)
      LLM_MAX_CHARS     = tostring(var.llm_max_chars)
      LLM_SEED          = tostring(var.llm_seed)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.pipeline_label_lambda_basic,
    aws_iam_role_policy.pipeline_label_lambda,
    aws_s3_bucket.data,
  ]
}

resource "aws_lambda_event_source_mapping" "label" {
  event_source_arn = aws_sqs_queue.label.arn
  function_name    = aws_lambda_function.pipeline_label.arn
  batch_size       = 1

  scaling_config {
    # LLM providers rate-limit aggressively; two workers is plenty for a daily run.
    maximum_concurrency = 2
  }
}

# ---------------------------------------------------------------------------
# cache_write: sole owner of DynamoDB sentiment_cache writes
# ---------------------------------------------------------------------------

resource "aws_iam_role" "cache_write_lambda" {
  name               = "${var.project_name}-cache-write-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "cache_write_lambda_basic" {
  role       = aws_iam_role.cache_write_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "cache_write_lambda" {
  name = "${var.project_name}-cache-write-perms"
  role = aws_iam_role.cache_write_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "WriteSentimentCache"
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem"]
        Resource = aws_dynamodb_table.sentiment_cache.arn
      },
      {
        Sid      = "ConsumeCacheWriteTasks"
        Effect   = "Allow"
        Action   = local.sqs_consume_actions
        Resource = aws_sqs_queue.cache_write.arn
      },
    ]
  })
}

resource "aws_lambda_function" "cache_write" {
  function_name = "${var.project_name}-cache-write"
  role          = aws_iam_role.cache_write_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.cache_write.repository_url}:${var.image_tag}"

  timeout     = 30
  memory_size = 128

  environment {
    variables = {
      TABLE_NAME = aws_dynamodb_table.sentiment_cache.name
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.cache_write_lambda_basic,
    aws_iam_role_policy.cache_write_lambda,
    aws_dynamodb_table.sentiment_cache,
  ]
}

resource "aws_lambda_event_source_mapping" "cache_write" {
  event_source_arn        = aws_sqs_queue.cache_write.arn
  function_name           = aws_lambda_function.cache_write.arn
  batch_size              = 10
  function_response_types = ["ReportBatchItemFailures"]

  maximum_batching_window_in_seconds = 5
}

# ---------------------------------------------------------------------------
# api_sentiment: real-time per-symbol sentiment API endpoint
# ---------------------------------------------------------------------------

resource "aws_iam_role" "api_sentiment_lambda" {
  name               = "${var.project_name}-api-sentiment-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "api_sentiment_lambda_basic" {
  role       = aws_iam_role.api_sentiment_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "api_sentiment_lambda" {
  name = "${var.project_name}-api-sentiment-perms"
  role = aws_iam_role.api_sentiment_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid      = "InvokeSageMakerEndpoint"
          Effect   = "Allow"
          Action   = ["sagemaker:InvokeEndpoint"]
          Resource = aws_sagemaker_endpoint.classifier.arn
        },
        {
          Sid      = "SendCacheWriteTasks"
          Effect   = "Allow"
          Action   = ["sqs:SendMessage"]
          Resource = aws_sqs_queue.cache_write.arn
        },
      ],
      var.reddit_credentials_secret_arn != "" ? [
        {
          Sid      = "ReadRedditSecret"
          Effect   = "Allow"
          Action   = ["secretsmanager:GetSecretValue"]
          Resource = var.reddit_credentials_secret_arn
        }
      ] : [],
      var.valid_tickers_ssm_param != "" ? [
        {
          Sid      = "ReadValidTickersParam"
          Effect   = "Allow"
          Action   = ["ssm:GetParameter"]
          Resource = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${var.valid_tickers_ssm_param}"
        }
      ] : []
    )
  })
}

resource "aws_lambda_function" "api_sentiment" {
  function_name = "${var.project_name}-api-sentiment"
  role          = aws_iam_role.api_sentiment_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api_sentiment.repository_url}:${var.image_tag}"

  timeout     = 29
  memory_size = 512

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME         = aws_sagemaker_endpoint.classifier.name
      REDDIT_SECRET_ARN               = var.reddit_credentials_secret_arn
      RECENT_HEADLINES_MAX            = "10"
      DEFAULT_MAX_ARTICLES            = "12"
      CACHE_WRITE_QUEUE_URL           = aws_sqs_queue.cache_write.url
      CACHE_TTL_SECONDS               = tostring(var.sentiment_cache_api_ttl_seconds)
      VALID_TICKERS_SSM_PARAM         = var.valid_tickers_ssm_param
      VALID_TICKERS_JSON              = var.valid_tickers_json
      VALID_TICKERS_FILE              = var.valid_tickers_file
      VALID_TICKERS_CACHE_TTL_SECONDS = tostring(var.valid_tickers_cache_ttl_seconds)
      RSS_OVERFETCH                   = tostring(var.rss_overfetch)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.api_sentiment_lambda_basic,
    aws_iam_role_policy.api_sentiment_lambda,
    aws_sagemaker_endpoint.classifier,
  ]
}

# ---------------------------------------------------------------------------
# api_cache_read: read-only view of the sentiment cache
# ---------------------------------------------------------------------------

resource "aws_iam_role" "api_cache_read_lambda" {
  name               = "${var.project_name}-api-cache-read-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "api_cache_read_lambda_basic" {
  role       = aws_iam_role.api_cache_read_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "api_cache_read_lambda" {
  name = "${var.project_name}-api-cache-read-perms"
  role = aws_iam_role.api_cache_read_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadSentimentCache"
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:Scan",
        ]
        Resource = aws_dynamodb_table.sentiment_cache.arn
      },
    ]
  })
}

resource "aws_lambda_function" "api_cache_read" {
  function_name = "${var.project_name}-api-cache-read"
  role          = aws_iam_role.api_cache_read_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api_cache_read.repository_url}:${var.image_tag}"

  timeout     = 10
  memory_size = 128

  environment {
    variables = {
      TABLE_NAME = aws_dynamodb_table.sentiment_cache.name
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.api_cache_read_lambda_basic,
    aws_iam_role_policy.api_cache_read_lambda,
    aws_dynamodb_table.sentiment_cache,
  ]
}

# ---------------------------------------------------------------------------
# api_ticker_suggest: ticker autocomplete over the valid-ticker universe
# ---------------------------------------------------------------------------

resource "aws_iam_role" "api_ticker_suggest_lambda" {
  name               = "${var.project_name}-api-ticker-suggest-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "api_ticker_suggest_lambda_basic" {
  role       = aws_iam_role.api_ticker_suggest_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "api_ticker_suggest_lambda" {
  count = var.valid_tickers_ssm_param != "" ? 1 : 0
  name  = "${var.project_name}-api-ticker-suggest-perms"
  role  = aws_iam_role.api_ticker_suggest_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadValidTickersParam"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${var.valid_tickers_ssm_param}"
      },
    ]
  })
}

resource "aws_lambda_function" "api_ticker_suggest" {
  function_name = "${var.project_name}-api-ticker-suggest"
  role          = aws_iam_role.api_ticker_suggest_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api_ticker_suggest.repository_url}:${var.image_tag}"

  timeout     = 10
  memory_size = 128

  environment {
    variables = {
      VALID_TICKERS_SSM_PARAM         = var.valid_tickers_ssm_param
      VALID_TICKERS_JSON              = var.valid_tickers_json
      VALID_TICKERS_FILE              = var.valid_tickers_file
      VALID_TICKERS_CACHE_TTL_SECONDS = tostring(var.valid_tickers_cache_ttl_seconds)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.api_ticker_suggest_lambda_basic,
  ]
}
