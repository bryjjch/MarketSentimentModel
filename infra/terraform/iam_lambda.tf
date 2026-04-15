data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "predict_lambda" {
  name               = "${var.project_name}-predict-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "predict_lambda_basic" {
  role       = aws_iam_role.predict_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "predict_lambda_invoke" {
  statement {
    sid       = "InvokeSageMakerEndpoint"
    actions   = ["sagemaker:InvokeEndpoint"]
    resources = [aws_sagemaker_endpoint.classifier.arn]
  }
}

resource "aws_iam_role_policy" "predict_lambda_invoke" {
  name   = "${var.project_name}-invoke-endpoint"
  role   = aws_iam_role.predict_lambda.id
  policy = data.aws_iam_policy_document.predict_lambda_invoke.json
}
