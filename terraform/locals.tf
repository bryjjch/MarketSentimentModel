locals {
  bucket_name = coalesce(
    var.bucket_name,
    "${var.project_name}-models-${data.aws_caller_identity.current.account_id}"
  )
  data_bucket_name = coalesce(
    var.data_bucket_name,
    "${var.project_name}-data-${data.aws_caller_identity.current.account_id}"
  )
  endpoint_name = coalesce(var.endpoint_name, "${var.project_name}-endpoint")
  pipeline_name = coalesce(var.pipeline_name, "${var.project_name}-training-pipeline")

  # The endpoint is created by model_promote, not Terraform, so its ARN is constructed
  # rather than referenced. Consumers only need it to scope InvokeEndpoint policies.
  endpoint_arn = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:endpoint/${local.endpoint_name}"
}
