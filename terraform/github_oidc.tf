# ---------------------------------------------------------------------------
# GitHub Actions OIDC: lets CI assume a role without any long-lived AWS keys.
#
# Bootstrap note: this file is the one chicken-and-egg in the setup. Apply it once
# from a workstation with admin credentials; every apply after that runs in CI.
# If the account already has a GitHub OIDC provider, import it instead of creating
# a second one (AWS permits only one per URL):
#   terraform import aws_iam_openid_connect_provider.github \
#     arn:aws:iam::<account>:oidc-provider/token.actions.githubusercontent.com
# ---------------------------------------------------------------------------

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]

  # IAM validates GitHub's certificate against its own trust store, so these values
  # are vestigial, but the API still requires the field.
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
  ]
}

# Shared assume-role document builder. `sub` is what pins each role to a specific
# trigger: without it, any repo on GitHub could assume these roles.
data "aws_iam_policy_document" "github_assume_plan" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Pull requests only. Plans run against untrusted PR code, so this role is read-only.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:pull_request"]
    }
  }
}

data "aws_iam_policy_document" "github_assume_apply" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Only the default branch. Covers both push-to-main and workflow_dispatch on main;
    # a PR branch cannot assume this even if the workflow file is modified in the PR.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:ref:refs/heads/${var.github_default_branch}"]
    }
  }
}

# ---------------------------------------------------------------------------
# Plan role (pull requests): read everything, plus the state lock.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "github_plan" {
  name               = "${var.project_name}-gha-plan"
  description        = "Assumed by GitHub Actions on pull requests to run terraform plan."
  assume_role_policy = data.aws_iam_policy_document.github_assume_plan.json
}

resource "aws_iam_role_policy_attachment" "github_plan_readonly" {
  role       = aws_iam_role.github_plan.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

data "aws_iam_policy_document" "github_tf_state" {
  statement {
    sid     = "StateObjectRead"
    actions = ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"]
    resources = [
      "arn:aws:s3:::${var.tf_state_bucket}",
      "arn:aws:s3:::${var.tf_state_bucket}/*",
    ]
  }

  # plan takes the lock too; without this every PR fails on a lock error.
  statement {
    sid       = "StateLock"
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"]
    resources = ["arn:aws:dynamodb:${var.aws_region}:${data.aws_caller_identity.current.account_id}:table/${var.tf_lock_table}"]
  }
}

resource "aws_iam_role_policy" "github_plan_state" {
  name   = "${var.project_name}-gha-plan-state"
  role   = aws_iam_role.github_plan.id
  policy = data.aws_iam_policy_document.github_tf_state.json
}

# ---------------------------------------------------------------------------
# Apply role (main): deploy the stack.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "github_apply" {
  name               = "${var.project_name}-gha-apply"
  description        = "Assumed by GitHub Actions on ${var.github_default_branch} to apply Terraform and push images."
  assume_role_policy = data.aws_iam_policy_document.github_assume_apply.json
}

# PowerUserAccess covers every service this stack touches (Lambda, ECR, S3, SQS,
# DynamoDB, API Gateway, EventBridge, SageMaker, SSM, Secrets Manager) but excludes
# IAM, which the inline policy below grants narrowly. Enumerating each service action
# instead would be a large, brittle policy that fails opaquely mid-apply; this trades
# some breadth for a role that only a push to main can assume.
resource "aws_iam_role_policy_attachment" "github_apply_poweruser" {
  role       = aws_iam_role.github_apply.name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

resource "aws_iam_role_policy" "github_apply_state" {
  name   = "${var.project_name}-gha-apply-state"
  role   = aws_iam_role.github_apply.id
  policy = data.aws_iam_policy_document.github_tf_state_write.json
}

data "aws_iam_policy_document" "github_tf_state_write" {
  statement {
    sid     = "StateObjectReadWrite"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket", "s3:GetBucketLocation"]
    resources = [
      "arn:aws:s3:::${var.tf_state_bucket}",
      "arn:aws:s3:::${var.tf_state_bucket}/*",
    ]
  }

  statement {
    sid       = "StateLock"
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"]
    resources = ["arn:aws:dynamodb:${var.aws_region}:${data.aws_caller_identity.current.account_id}:table/${var.tf_lock_table}"]
  }
}

# IAM, scoped to the roles and policies this stack owns. PowerUserAccess grants none
# of this, so Terraform cannot touch identities outside the project's naming prefix.
data "aws_iam_policy_document" "github_apply_iam" {
  statement {
    sid = "ManageProjectRoles"
    actions = [
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:GetRole",
      "iam:UpdateRole",
      "iam:UpdateRoleDescription",
      "iam:UpdateAssumeRolePolicy",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:ListInstanceProfilesForRole",
      "iam:PutRolePolicy",
      "iam:GetRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:TagRole",
      "iam:UntagRole",
      "iam:ListRoleTags",
    ]
    resources = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project_name}-*"]
  }

  statement {
    sid       = "PassProjectRoles"
    actions   = ["iam:PassRole"]
    resources = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project_name}-*"]
  }

  # Read-only on the OIDC provider so `plan` shows no drift on this file; the provider
  # itself stays a bootstrap-only resource that CI cannot delete out from under itself.
  statement {
    sid       = "ReadOIDCProvider"
    actions   = ["iam:GetOpenIDConnectProvider", "iam:ListOpenIDConnectProviders"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_apply_iam" {
  name   = "${var.project_name}-gha-apply-iam"
  role   = aws_iam_role.github_apply.id
  policy = data.aws_iam_policy_document.github_apply_iam.json
}

# ---------------------------------------------------------------------------
# Outputs consumed when configuring the GitHub repository
# ---------------------------------------------------------------------------

output "github_plan_role_arn" {
  description = "Set as the GitHub Actions repository variable AWS_PLAN_ROLE_ARN."
  value       = aws_iam_role.github_plan.arn
}

output "github_apply_role_arn" {
  description = "Set as the GitHub Actions repository variable AWS_APPLY_ROLE_ARN."
  value       = aws_iam_role.github_apply.arn
}
