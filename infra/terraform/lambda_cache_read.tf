data "archive_file" "cache_read_lambda" {
  type        = "zip"
  source_file = abspath("${path.module}/../lambda/cache_read/handler.py")
  output_path = "${path.module}/build/cache_read.zip"
}

resource "aws_lambda_function" "cache_read" {
  function_name = "${var.project_name}-sentiment-cache-read"
  role          = aws_iam_role.cache_read_lambda.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.12"

  filename         = data.archive_file.cache_read_lambda.output_path
  source_code_hash = data.archive_file.cache_read_lambda.output_base64sha256
  layers = [
    aws_lambda_layer_version.finsense_shared.arn,
    aws_lambda_layer_version.finsense_deps.arn,
  ]

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
    aws_lambda_layer_version.finsense_shared,
    aws_lambda_layer_version.finsense_deps,
  ]
}
