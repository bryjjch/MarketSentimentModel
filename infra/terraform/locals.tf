locals {
  # Reddit + Finnhub credentials (Secrets Manager); empty strings omitted.
  provider_secret_arns = compact([
    var.reddit_credentials_secret_arn,
    var.finnhub_secret_arn,
  ])
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
}
