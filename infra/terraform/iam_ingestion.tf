resource "aws_iam_role" "ingestion_lambda" {
  name               = "${var.project_name}-ingestion-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "ingestion_lambda_basic" {
  role       = aws_iam_role.ingestion_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "ingestion_lambda" {
  name = "${var.project_name}-ingestion-perms"
  role = aws_iam_role.ingestion_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid    = "WriteRawData"
          Effect = "Allow"
          Action = [
            "s3:PutObject",
            "s3:AbortMultipartUpload",
          ]
          Resource = "${aws_s3_bucket.data.arn}/raw/*"
        },
        {
          Sid      = "ListDataBucket"
          Effect   = "Allow"
          Action   = ["s3:ListBucket"]
          Resource = aws_s3_bucket.data.arn
        },
        {
          Sid      = "InvokeIngestionPredictionLambda"
          Effect   = "Allow"
          Action   = ["lambda:InvokeFunction"]
          Resource = aws_lambda_function.ingestion_prediction.arn
        },
        {
          Sid      = "ReadTickerParam"
          Effect   = "Allow"
          Action   = ["ssm:GetParameter"]
          Resource = aws_ssm_parameter.top_tickers.arn
        },
      ],
      length(local.provider_secret_arns) > 0 ? [
        {
          Sid      = "ReadProviderSecrets"
          Effect   = "Allow"
          Action   = ["secretsmanager:GetSecretValue"]
          Resource = local.provider_secret_arns
        }
      ] : []
    )
  })
}
