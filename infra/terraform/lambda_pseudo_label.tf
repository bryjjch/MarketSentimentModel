data "archive_file" "pseudo_label_lambda" {
  type        = "zip"
  source_dir  = abspath("${path.module}/../lambda/pseudo_label")
  output_path = "${path.module}/build/pseudo_label.zip"
  excludes    = ["__pycache__", "*.pyc", ".pytest_cache"]
}

resource "aws_lambda_function" "pseudo_label" {
  function_name = "${var.project_name}-pseudo-label"
  role          = aws_iam_role.pseudo_label_lambda.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.12"

  filename         = data.archive_file.pseudo_label_lambda.output_path
  source_code_hash = data.archive_file.pseudo_label_lambda.output_base64sha256

  layers = [
    aws_lambda_layer_version.finsense_shared.arn,
    aws_lambda_layer_version.finsense_deps.arn,
  ]

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
