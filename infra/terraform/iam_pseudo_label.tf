resource "aws_iam_role" "pseudo_label_lambda" {
  name               = "${var.project_name}-pseudo-label-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "pseudo_label_lambda_basic" {
  role       = aws_iam_role.pseudo_label_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

locals {
  llm_secret_arns = compact([var.openai_secret_arn, var.google_secret_arn])
}

resource "aws_iam_role_policy" "pseudo_label_lambda" {
  name = "${var.project_name}-pseudo-label-perms"
  role = aws_iam_role.pseudo_label_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid      = "ReadPredictions"
          Effect   = "Allow"
          Action   = ["s3:GetObject"]
          Resource = "${aws_s3_bucket.data.arn}/predictions/*"
        },
        {
          Sid    = "WritePseudoAndCurated"
          Effect = "Allow"
          Action = [
            "s3:PutObject",
            "s3:AbortMultipartUpload",
          ]
          Resource = [
            "${aws_s3_bucket.data.arn}/pseudo/*",
            "${aws_s3_bucket.data.arn}/curated/*",
          ]
        },
        {
          Sid      = "ListDataBucket"
          Effect   = "Allow"
          Action   = ["s3:ListBucket"]
          Resource = aws_s3_bucket.data.arn
        },
      ],
      length(local.llm_secret_arns) > 0 ? [
        {
          Sid      = "ReadLLMSecrets"
          Effect   = "Allow"
          Action   = ["secretsmanager:GetSecretValue"]
          Resource = local.llm_secret_arns
        }
      ] : []
    )
  })
}
