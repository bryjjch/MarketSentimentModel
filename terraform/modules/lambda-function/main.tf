# A container-image Lambda plus the IAM role it runs as.
#
# Every FinSense Lambda has the same shape — role, AWSLambdaBasicExecutionRole, one
# inline policy, one image-backed function — so the callers only describe what differs.
# Names are derived from var.name so they keep matching the deployed resources:
#   function  <name>
#   role      <name>-lambda
#   policy    <name>-perms

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "this" {
  name               = "${var.name}-lambda"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy_attachment" "basic" {
  role       = aws_iam_role.this.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Optional: a function whose only permissions are CloudWatch Logs passes policy_json = null.
resource "aws_iam_role_policy" "this" {
  count = var.policy_json == null ? 0 : 1

  name   = "${var.name}-perms"
  role   = aws_iam_role.this.id
  policy = var.policy_json
}

resource "aws_lambda_function" "this" {
  function_name = var.name
  role          = aws_iam_role.this.arn
  package_type  = "Image"
  image_uri     = var.image_uri

  timeout     = var.timeout
  memory_size = var.memory_size

  dynamic "environment" {
    for_each = length(var.environment) > 0 ? [1] : []
    content {
      variables = var.environment
    }
  }

  # Without this the function can be created before its permissions exist, and a
  # cold start that lands in that window fails on AccessDenied.
  depends_on = [
    aws_iam_role_policy_attachment.basic,
    aws_iam_role_policy.this,
  ]
}
