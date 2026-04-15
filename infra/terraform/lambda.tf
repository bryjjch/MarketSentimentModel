data "archive_file" "predict_lambda" {
  type        = "zip"
  source_file = abspath("${path.module}/../lambda/predict/handler.py")
  output_path = "${path.module}/build/predict.zip"
}

resource "aws_lambda_function" "predict" {
  function_name = "${var.project_name}-predict"
  role          = aws_iam_role.predict_lambda.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.12"

  filename         = data.archive_file.predict_lambda.output_path
  source_code_hash = data.archive_file.predict_lambda.output_base64sha256

  timeout     = 29
  memory_size = 256

  reserved_concurrent_executions = var.lambda_reserved_concurrent_executions

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = aws_sagemaker_endpoint.classifier.name
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.predict_lambda_basic,
    aws_iam_role_policy.predict_lambda_invoke,
    aws_sagemaker_endpoint.classifier,
  ]
}
