resource "aws_iam_role" "sentiment_refresh_lambda" {
  name               = "${var.project_name}-sentiment-refresh-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "sentiment_refresh_lambda_basic" {
  role       = aws_iam_role.sentiment_refresh_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "sentiment_refresh_invoke" {
  name = "${var.project_name}-sentiment-refresh-perms"
  role = aws_iam_role.sentiment_refresh_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction",
        ]
        Resource = aws_lambda_function.sentiment.arn
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
        ]
        Resource = aws_dynamodb_table.sentiment_cache.arn
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
        ]
        Resource = aws_ssm_parameter.top_tickers.arn
      },
    ]
  })
}
