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
        ]
        Resource = aws_dynamodb_table.sentiment_cache.arn
      },
    ]
  })
}
