resource "aws_iam_role" "sentiment_lambda" {
  name               = "${var.project_name}-sentiment-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "sentiment_lambda_basic" {
  role       = aws_iam_role.sentiment_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "sentiment_lambda_invoke" {
  name = "${var.project_name}-sentiment-invoke-sm"
  role = aws_iam_role.sentiment_lambda.id

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

resource "aws_iam_role_policy" "sentiment_lambda_secrets" {
  count = var.reddit_credentials_secret_arn != "" ? 1 : 0
  name  = "${var.project_name}-sentiment-read-reddit-secret"
  role  = aws_iam_role.sentiment_lambda.id

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
