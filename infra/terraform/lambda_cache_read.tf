resource "aws_lambda_function" "cache_read" {
  function_name = "${var.project_name}-sentiment-cache-read"
  role          = aws_iam_role.cache_read_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.cache_read.repository_url}:latest"

  timeout     = 10
  memory_size = 128

  environment {
    variables = {
      TABLE_NAME = aws_dynamodb_table.sentiment_cache.name
      VALID_TICKERS_SSM_PARAM = var.valid_tickers_ssm_param
      VALID_TICKERS_JSON      = var.valid_tickers_json
      VALID_TICKERS_FILE      = var.valid_tickers_file
      VALID_TICKERS_CACHE_TTL_SECONDS = tostring(var.valid_tickers_cache_ttl_seconds)
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.cache_read_lambda_basic,
    aws_iam_role_policy.cache_read_ddb,
    aws_dynamodb_table.sentiment_cache,
  ]
}
