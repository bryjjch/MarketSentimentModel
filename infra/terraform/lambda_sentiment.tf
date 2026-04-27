data "archive_file" "sentiment_lambda" {
  type        = "zip"
  source_dir  = abspath("${path.module}/../lambda/sentiment")
  output_path = "${path.module}/build/sentiment.zip"
  excludes    = ["__pycache__", "*.pyc", ".pytest_cache"]
}

resource "aws_lambda_function" "sentiment" {
  function_name = "${var.project_name}-sentiment"
  role            = aws_iam_role.sentiment_lambda.arn
  handler         = "handler.lambda_handler"
  runtime         = "python3.12"

  filename         = data.archive_file.sentiment_lambda.output_path
  source_code_hash = data.archive_file.sentiment_lambda.output_base64sha256
  layers           = [aws_lambda_layer_version.finsense_shared.arn]

  timeout     = 29
  memory_size = 512

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = aws_sagemaker_endpoint.classifier.name
      REDDIT_SECRET_ARN = var.reddit_credentials_secret_arn
      RECENT_HEADLINES_MAX = "10"
      DEFAULT_MAX_ARTICLES = "12"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.sentiment_lambda_basic,
    aws_iam_role_policy.sentiment_lambda_invoke,
    aws_sagemaker_endpoint.classifier,
  ]
}
