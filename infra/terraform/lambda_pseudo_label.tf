resource "aws_lambda_function" "pseudo_label" {
  function_name = "${var.project_name}-pseudo-label"
  role          = aws_iam_role.pseudo_label_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.pseudo_label.repository_url}:${var.image_tag}"

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
