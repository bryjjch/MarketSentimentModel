resource "aws_lambda_function" "ingestion" {
  function_name = "${var.project_name}-ingestion"
  role          = aws_iam_role.ingestion_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.ingestion.repository_url}:latest"

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
