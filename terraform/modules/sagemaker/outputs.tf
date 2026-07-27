output "pipeline_role_arn" {
  description = "IAM role ARN used by the SageMaker Pipeline. Passed to build_pipeline.py --role."
  value       = aws_iam_role.pipeline.arn
}

output "model_package_group_name" {
  description = "Model Package Group for registered model versions."
  value       = aws_sagemaker_model_package_group.this.model_package_group_name
}
