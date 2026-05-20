resource "aws_lambda_function" "api_inference" {
  function_name = "${var.project_name}-api-inference"
  role          = aws_iam_role.api_inference_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api_inference.repository_url}:latest"

  timeout     = 29
  memory_size = 256

  reserved_concurrent_executions = var.lambda_reserved_concurrent_executions

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = aws_sagemaker_endpoint.classifier.name
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.api_inference_lambda_basic,
    aws_iam_role_policy.api_inference_lambda_invoke,
    aws_sagemaker_endpoint.classifier,
  ]
}
