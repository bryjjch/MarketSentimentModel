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

# ---------------------------------------------------------------------------
# api_inference: invokes the SageMaker classifier endpoint
# ---------------------------------------------------------------------------

resource "aws_iam_role" "api_inference_lambda" {
  name               = "${var.project_name}-api-inference-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "api_inference_lambda_basic" {
  role       = aws_iam_role.api_inference_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "api_inference_lambda_invoke" {
  statement {
    sid       = "InvokeSageMakerEndpoint"
    actions   = ["sagemaker:InvokeEndpoint"]
    resources = [aws_sagemaker_endpoint.classifier.arn]
  }
}

resource "aws_iam_role_policy" "api_inference_lambda_invoke" {
  name   = "${var.project_name}-invoke-endpoint"
  role   = aws_iam_role.api_inference_lambda.id
  policy = data.aws_iam_policy_document.api_inference_lambda_invoke.json
}

resource "aws_lambda_function" "api_inference" {
  function_name = "${var.project_name}-api-inference"
  role          = aws_iam_role.api_inference_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api_inference.repository_url}:${var.image_tag}"

  timeout     = 29
  memory_size = 256

  reserved_concurrent_executions = var.lambda_reserved_concurrent_executions

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = aws_sagemaker_endpoint.classifier.name
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.api_inference_lambda_basic,
    aws_iam_role_policy.api_inference_lambda_invoke,
    aws_sagemaker_endpoint.classifier,
  ]
}

# ---------------------------------------------------------------------------
# cache_read: reads precomputed sentiment from DynamoDB
# ---------------------------------------------------------------------------

resource "aws_iam_role" "cache_read_lambda" {
  name               = "${var.project_name}-sentiment-cache-read-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "cache_read_lambda_basic" {
  role       = aws_iam_role.cache_read_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "cache_read_ddb" {
  name = "${var.project_name}-cache-read-ddb"
  role = aws_iam_role.cache_read_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
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

resource "aws_iam_role_policy" "cache_read_valid_tickers_ssm" {
  count = var.valid_tickers_ssm_param != "" ? 1 : 0
  name  = "${var.project_name}-cache-read-valid-tickers-ssm"
  role  = aws_iam_role.cache_read_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["ssm:GetParameter"]
        Resource = [
          "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${var.valid_tickers_ssm_param}",
        ]
      },
    ]
  })
}

resource "aws_lambda_function" "cache_read" {
  function_name = "${var.project_name}-sentiment-cache-read"
  role          = aws_iam_role.cache_read_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.cache_read.repository_url}:${var.image_tag}"

  timeout     = 10
  memory_size = 128

  environment {
    variables = {
      TABLE_NAME                      = aws_dynamodb_table.sentiment_cache.name
      VALID_TICKERS_SSM_PARAM         = var.valid_tickers_ssm_param
      VALID_TICKERS_JSON              = var.valid_tickers_json
      VALID_TICKERS_FILE              = var.valid_tickers_file
      VALID_TICKERS_CACHE_TTL_SECONDS = tostring(var.valid_tickers_cache_ttl_seconds)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.cache_read_lambda_basic,
    aws_iam_role_policy.cache_read_ddb,
    aws_dynamodb_table.sentiment_cache,
  ]
}

# ---------------------------------------------------------------------------
# ingestion: daily fan-out triggered by EventBridge
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ingestion_lambda" {
  name               = "${var.project_name}-ingestion-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "ingestion_lambda_basic" {
  role       = aws_iam_role.ingestion_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "ingestion_lambda" {
  name = "${var.project_name}-ingestion-perms"
  role = aws_iam_role.ingestion_lambda.id

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
          Sid      = "InvokeIngestionPredictionLambda"
          Effect   = "Allow"
          Action   = ["lambda:InvokeFunction"]
          Resource = aws_lambda_function.ingestion_prediction.arn
        },
        {
          Sid      = "ReadTickerParam"
          Effect   = "Allow"
          Action   = ["ssm:GetParameter"]
          Resource = aws_ssm_parameter.top_tickers.arn
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

resource "aws_lambda_function" "ingestion" {
  function_name = "${var.project_name}-ingestion"
  role          = aws_iam_role.ingestion_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.ingestion.repository_url}:${var.image_tag}"

  timeout     = 600
  memory_size = var.ingestion_lambda_memory_mb

  environment {
    variables = {
      DATA_BUCKET                        = aws_s3_bucket.data.bucket
      INGESTION_PREDICTION_FUNCTION_NAME = aws_lambda_function.ingestion_prediction.function_name
      DEFAULT_MAX_ARTICLES               = tostring(var.ingestion_max_articles)
      INCLUDE_SOCIAL                     = var.ingestion_include_social ? "true" : "false"
      TOP_TICKERS_SSM_PARAM              = aws_ssm_parameter.top_tickers.name
      DEFAULT_TICKERS_JSON               = var.top_tickers_json
      REDDIT_SECRET_ARN                  = var.reddit_credentials_secret_arn
      RSS_OVERFETCH                      = tostring(var.rss_overfetch)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.ingestion_lambda_basic,
    aws_iam_role_policy.ingestion_lambda,
    aws_s3_bucket.data,
    aws_lambda_function.ingestion_prediction,
    aws_ssm_parameter.top_tickers,
  ]
}

# ---------------------------------------------------------------------------
# ingestion_prediction: per-ticker prediction invoked by ingestion fan-out
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ingestion_prediction_lambda" {
  name               = "${var.project_name}-ingestion-prediction-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "ingestion_prediction_lambda_basic" {
  role       = aws_iam_role.ingestion_prediction_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "ingestion_prediction_lambda" {
  name = "${var.project_name}-ingestion-prediction-perms"
  role = aws_iam_role.ingestion_prediction_lambda.id

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
        Sid      = "InvokePseudoLabelLambda"
        Effect   = "Allow"
        Action   = ["lambda:InvokeFunction"]
        Resource = aws_lambda_function.pseudo_label.arn
      },
      {
        Sid      = "WriteSentimentCache"
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem"]
        Resource = aws_dynamodb_table.sentiment_cache.arn
      },
    ]
  })
}

resource "aws_lambda_function" "ingestion_prediction" {
  function_name = "${var.project_name}-ingestion-prediction"
  role          = aws_iam_role.ingestion_prediction_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.ingestion_prediction.repository_url}:${var.image_tag}"

  timeout     = 600
  memory_size = var.prediction_lambda_memory_mb

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME    = aws_sagemaker_endpoint.classifier.name
      DATA_BUCKET                = aws_s3_bucket.data.bucket
      PSEUDO_LABEL_FUNCTION_NAME = aws_lambda_function.pseudo_label.function_name
      CACHE_TABLE_NAME           = aws_dynamodb_table.sentiment_cache.name
      CACHE_TTL_SECONDS          = tostring(var.sentiment_cache_ttl_seconds)
      RECENT_HEADLINES_MAX       = "10"
      LOW_CONF_TOP_PROB          = tostring(var.low_conf_top_prob)
      LOW_CONF_MARGIN            = tostring(var.low_conf_margin)
      SAGEMAKER_BATCH_SIZE       = tostring(var.sagemaker_batch_size)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.ingestion_prediction_lambda_basic,
    aws_iam_role_policy.ingestion_prediction_lambda,
    aws_s3_bucket.data,
    aws_sagemaker_endpoint.classifier,
    aws_lambda_function.pseudo_label,
    aws_dynamodb_table.sentiment_cache,
  ]
}

# ---------------------------------------------------------------------------
# pseudo_label: LLM-backed labeler for low-confidence predictions
# ---------------------------------------------------------------------------

locals {
  llm_secret_arns = compact([var.openai_secret_arn, var.google_secret_arn])
}

resource "aws_iam_role" "pseudo_label_lambda" {
  name               = "${var.project_name}-pseudo-label-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "pseudo_label_lambda_basic" {
  role       = aws_iam_role.pseudo_label_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "pseudo_label_lambda" {
  name = "${var.project_name}-pseudo-label-perms"
  role = aws_iam_role.pseudo_label_lambda.id

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

resource "aws_lambda_function" "pseudo_label" {
  function_name = "${var.project_name}-pseudo-label"
  role          = aws_iam_role.pseudo_label_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.pseudo_label.repository_url}:${var.image_tag}"

  timeout     = 600
  memory_size = var.pseudo_label_lambda_memory_mb

  environment {
    variables = {
      DATA_BUCKET       = aws_s3_bucket.data.bucket
      LLM_PROVIDER      = var.llm_provider
      LLM_MODEL         = var.llm_model
      OPENAI_SECRET_ARN = var.openai_secret_arn
      GOOGLE_SECRET_ARN = var.google_secret_arn
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.pseudo_label_lambda_basic,
    aws_iam_role_policy.pseudo_label_lambda,
    aws_s3_bucket.data,
  ]
}

# ---------------------------------------------------------------------------
# api_sentiment_by_symbol: real-time per-symbol sentiment API endpoint
# ---------------------------------------------------------------------------

resource "aws_iam_role" "api_sentiment_by_symbol_lambda" {
  name               = "${var.project_name}-api-sentiment-by-symbol-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "api_sentiment_by_symbol_lambda_basic" {
  role       = aws_iam_role.api_sentiment_by_symbol_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "api_sentiment_by_symbol_lambda_invoke" {
  name = "${var.project_name}-api-sentiment-by-symbol-invoke-sm"
  role = aws_iam_role.api_sentiment_by_symbol_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "InvokeSageMakerEndpoint"
        Effect   = "Allow"
        Action   = ["sagemaker:InvokeEndpoint"]
        Resource = aws_sagemaker_endpoint.classifier.arn
      },
    ]
  })
}

resource "aws_iam_role_policy" "api_sentiment_by_symbol_lambda_cache_write" {
  name = "${var.project_name}-api-sentiment-by-symbol-cache-write"
  role = aws_iam_role.api_sentiment_by_symbol_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "WriteSentimentCache"
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem"]
        Resource = aws_dynamodb_table.sentiment_cache.arn
      },
    ]
  })
}

resource "aws_iam_role_policy" "api_sentiment_by_symbol_lambda_secrets" {
  count = var.reddit_credentials_secret_arn != "" ? 1 : 0
  name  = "${var.project_name}-api-sentiment-by-symbol-read-reddit-secret"
  role  = aws_iam_role.api_sentiment_by_symbol_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = [
          var.reddit_credentials_secret_arn,
        ]
      },
    ]
  })
}

resource "aws_iam_role_policy" "api_sentiment_by_symbol_lambda_valid_tickers_ssm" {
  count = var.valid_tickers_ssm_param != "" ? 1 : 0
  name  = "${var.project_name}-api-sentiment-by-symbol-valid-tickers-ssm"
  role  = aws_iam_role.api_sentiment_by_symbol_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["ssm:GetParameter"]
        Resource = [
          "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${var.valid_tickers_ssm_param}",
        ]
      },
    ]
  })
}

resource "aws_lambda_function" "api_sentiment_by_symbol" {
  function_name = "${var.project_name}-api-sentiment-by-symbol"
  role          = aws_iam_role.api_sentiment_by_symbol_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api_sentiment_by_symbol.repository_url}:${var.image_tag}"

  timeout     = 29
  memory_size = 512

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME         = aws_sagemaker_endpoint.classifier.name
      REDDIT_SECRET_ARN               = var.reddit_credentials_secret_arn
      RECENT_HEADLINES_MAX            = "10"
      DEFAULT_MAX_ARTICLES            = "12"
      CACHE_TABLE_NAME                = aws_dynamodb_table.sentiment_cache.name
      CACHE_TTL_SECONDS               = tostring(var.sentiment_cache_api_ttl_seconds)
      VALID_TICKERS_SSM_PARAM         = var.valid_tickers_ssm_param
      VALID_TICKERS_JSON              = var.valid_tickers_json
      VALID_TICKERS_FILE              = var.valid_tickers_file
      VALID_TICKERS_CACHE_TTL_SECONDS = tostring(var.valid_tickers_cache_ttl_seconds)
      RSS_OVERFETCH                   = tostring(var.rss_overfetch)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.api_sentiment_by_symbol_lambda_basic,
    aws_iam_role_policy.api_sentiment_by_symbol_lambda_invoke,
    aws_iam_role_policy.api_sentiment_by_symbol_lambda_cache_write,
    aws_dynamodb_table.sentiment_cache,
    aws_sagemaker_endpoint.classifier,
  ]
}
