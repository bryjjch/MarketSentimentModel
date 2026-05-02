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
  count = length(local.provider_secret_arns) > 0 ? 1 : 0
  name  = "${var.project_name}-api-sentiment-by-symbol-read-provider-secrets"
  role  = aws_iam_role.api_sentiment_by_symbol_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = local.provider_secret_arns
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
