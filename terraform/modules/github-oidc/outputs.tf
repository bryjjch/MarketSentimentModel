output "plan_role_arn" {
  description = "Role GitHub Actions assumes on pull requests."
  value       = aws_iam_role.plan.arn
}

output "apply_role_arn" {
  description = "Role GitHub Actions assumes on the default branch."
  value       = aws_iam_role.apply.arn
}
