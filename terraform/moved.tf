# State migration for the flat-files -> modules reorganisation.
#
# Every resource below kept its AWS identity; only its Terraform address changed. These
# blocks tell Terraform to rewrite state rather than destroy and recreate, so the first
# plan after this refactor shows no changes.
#
# Safe to delete once applied against every environment holding state (prod is the only
# one today). Until then, removing a block would orphan the resource it names.

# ---------------------------------------------------------------------------
# ECR: one resource per repository -> a single for_each
# ---------------------------------------------------------------------------

moved {
  from = aws_ecr_repository.pipeline_dispatch
  to   = module.ecr.aws_ecr_repository.this["pipeline-dispatch"]
}

moved {
  from = aws_ecr_repository.pipeline_collect
  to   = module.ecr.aws_ecr_repository.this["pipeline-collect"]
}

moved {
  from = aws_ecr_repository.pipeline_predict
  to   = module.ecr.aws_ecr_repository.this["pipeline-predict"]
}

moved {
  from = aws_ecr_repository.pipeline_label
  to   = module.ecr.aws_ecr_repository.this["pipeline-label"]
}

moved {
  from = aws_ecr_repository.cache_write
  to   = module.ecr.aws_ecr_repository.this["cache-write"]
}

moved {
  from = aws_ecr_repository.api_sentiment
  to   = module.ecr.aws_ecr_repository.this["api-sentiment"]
}

moved {
  from = aws_ecr_repository.api_cache_read
  to   = module.ecr.aws_ecr_repository.this["api-cache-read"]
}

moved {
  from = aws_ecr_repository.api_ticker_suggest
  to   = module.ecr.aws_ecr_repository.this["api-ticker-suggest"]
}

moved {
  from = aws_ecr_repository.model_promote
  to   = module.ecr.aws_ecr_repository.this["model-promote"]
}

# ---------------------------------------------------------------------------
# Storage: buckets share the s3-bucket module; table and lifecycle stay in storage
# ---------------------------------------------------------------------------

moved {
  from = aws_s3_bucket.models
  to   = module.storage.module.models.aws_s3_bucket.this
}

moved {
  from = aws_s3_bucket_public_access_block.models
  to   = module.storage.module.models.aws_s3_bucket_public_access_block.this
}

moved {
  from = aws_s3_bucket_ownership_controls.models
  to   = module.storage.module.models.aws_s3_bucket_ownership_controls.this
}

moved {
  from = aws_s3_bucket_server_side_encryption_configuration.models
  to   = module.storage.module.models.aws_s3_bucket_server_side_encryption_configuration.this
}

moved {
  from = aws_s3_bucket_versioning.models
  to   = module.storage.module.models.aws_s3_bucket_versioning.this
}

moved {
  from = aws_s3_bucket_policy.models_tls
  to   = module.storage.module.models.aws_s3_bucket_policy.tls
}

moved {
  from = aws_s3_bucket.data
  to   = module.storage.module.data.aws_s3_bucket.this
}

moved {
  from = aws_s3_bucket_public_access_block.data
  to   = module.storage.module.data.aws_s3_bucket_public_access_block.this
}

moved {
  from = aws_s3_bucket_ownership_controls.data
  to   = module.storage.module.data.aws_s3_bucket_ownership_controls.this
}

moved {
  from = aws_s3_bucket_server_side_encryption_configuration.data
  to   = module.storage.module.data.aws_s3_bucket_server_side_encryption_configuration.this
}

moved {
  from = aws_s3_bucket_versioning.data
  to   = module.storage.module.data.aws_s3_bucket_versioning.this
}

moved {
  from = aws_s3_bucket_policy.data_tls
  to   = module.storage.module.data.aws_s3_bucket_policy.tls
}

moved {
  from = aws_s3_bucket_lifecycle_configuration.data
  to   = module.storage.aws_s3_bucket_lifecycle_configuration.data
}

moved {
  from = aws_dynamodb_table.sentiment_cache
  to   = module.storage.aws_dynamodb_table.sentiment_cache
}

# The ticker parameter moved to the pipeline module, next to the Lambda that reads it.
moved {
  from = aws_ssm_parameter.top_tickers
  to   = module.pipeline.aws_ssm_parameter.top_tickers
}

# ---------------------------------------------------------------------------
# SQS: one resource per queue -> for_each over queue settings
# ---------------------------------------------------------------------------

moved {
  from = aws_sqs_queue.collect
  to   = module.queues.aws_sqs_queue.this["collect"]
}

moved {
  from = aws_sqs_queue.collect_dlq
  to   = module.queues.aws_sqs_queue.dlq["collect"]
}

moved {
  from = aws_sqs_queue.predict
  to   = module.queues.aws_sqs_queue.this["predict"]
}

moved {
  from = aws_sqs_queue.predict_dlq
  to   = module.queues.aws_sqs_queue.dlq["predict"]
}

moved {
  from = aws_sqs_queue.label
  to   = module.queues.aws_sqs_queue.this["label"]
}

moved {
  from = aws_sqs_queue.label_dlq
  to   = module.queues.aws_sqs_queue.dlq["label"]
}

moved {
  from = aws_sqs_queue.cache_write
  to   = module.queues.aws_sqs_queue.this["cache_write"]
}

moved {
  from = aws_sqs_queue.cache_write_dlq
  to   = module.queues.aws_sqs_queue.dlq["cache_write"]
}

# ---------------------------------------------------------------------------
# Pipeline Lambdas: each function's role/policy/function now come from
# modules/lambda-function, so the inline policy picks up a [0] index from its count.
# ---------------------------------------------------------------------------

moved {
  from = aws_iam_role.pipeline_dispatch_lambda
  to   = module.pipeline.module.dispatch.aws_iam_role.this
}

moved {
  from = aws_iam_role_policy_attachment.pipeline_dispatch_lambda_basic
  to   = module.pipeline.module.dispatch.aws_iam_role_policy_attachment.basic
}

moved {
  from = aws_iam_role_policy.pipeline_dispatch_lambda
  to   = module.pipeline.module.dispatch.aws_iam_role_policy.this[0]
}

moved {
  from = aws_lambda_function.pipeline_dispatch
  to   = module.pipeline.module.dispatch.aws_lambda_function.this
}

moved {
  from = aws_iam_role.pipeline_collect_lambda
  to   = module.pipeline.module.collect.aws_iam_role.this
}

moved {
  from = aws_iam_role_policy_attachment.pipeline_collect_lambda_basic
  to   = module.pipeline.module.collect.aws_iam_role_policy_attachment.basic
}

moved {
  from = aws_iam_role_policy.pipeline_collect_lambda
  to   = module.pipeline.module.collect.aws_iam_role_policy.this[0]
}

moved {
  from = aws_lambda_function.pipeline_collect
  to   = module.pipeline.module.collect.aws_lambda_function.this
}

moved {
  from = aws_iam_role.pipeline_predict_lambda
  to   = module.pipeline.module.predict.aws_iam_role.this
}

moved {
  from = aws_iam_role_policy_attachment.pipeline_predict_lambda_basic
  to   = module.pipeline.module.predict.aws_iam_role_policy_attachment.basic
}

moved {
  from = aws_iam_role_policy.pipeline_predict_lambda
  to   = module.pipeline.module.predict.aws_iam_role_policy.this[0]
}

moved {
  from = aws_lambda_function.pipeline_predict
  to   = module.pipeline.module.predict.aws_lambda_function.this
}

moved {
  from = aws_iam_role.pipeline_label_lambda
  to   = module.pipeline.module.label.aws_iam_role.this
}

moved {
  from = aws_iam_role_policy_attachment.pipeline_label_lambda_basic
  to   = module.pipeline.module.label.aws_iam_role_policy_attachment.basic
}

moved {
  from = aws_iam_role_policy.pipeline_label_lambda
  to   = module.pipeline.module.label.aws_iam_role_policy.this[0]
}

moved {
  from = aws_lambda_function.pipeline_label
  to   = module.pipeline.module.label.aws_lambda_function.this
}

moved {
  from = aws_iam_role.cache_write_lambda
  to   = module.pipeline.module.cache_write.aws_iam_role.this
}

moved {
  from = aws_iam_role_policy_attachment.cache_write_lambda_basic
  to   = module.pipeline.module.cache_write.aws_iam_role_policy_attachment.basic
}

moved {
  from = aws_iam_role_policy.cache_write_lambda
  to   = module.pipeline.module.cache_write.aws_iam_role_policy.this[0]
}

moved {
  from = aws_lambda_function.cache_write
  to   = module.pipeline.module.cache_write.aws_lambda_function.this
}

moved {
  from = aws_lambda_event_source_mapping.collect
  to   = module.pipeline.aws_lambda_event_source_mapping.collect
}

moved {
  from = aws_lambda_event_source_mapping.predict
  to   = module.pipeline.aws_lambda_event_source_mapping.predict
}

moved {
  from = aws_lambda_event_source_mapping.label
  to   = module.pipeline.aws_lambda_event_source_mapping.label
}

moved {
  from = aws_lambda_event_source_mapping.cache_write
  to   = module.pipeline.aws_lambda_event_source_mapping.cache_write
}

moved {
  from = aws_cloudwatch_event_rule.pipeline_dispatch
  to   = module.pipeline.aws_cloudwatch_event_rule.dispatch
}

moved {
  from = aws_cloudwatch_event_target.pipeline_dispatch
  to   = module.pipeline.aws_cloudwatch_event_target.dispatch
}

moved {
  from = aws_lambda_permission.eventbridge_invoke_pipeline_dispatch
  to   = module.pipeline.aws_lambda_permission.dispatch_events
}

# ---------------------------------------------------------------------------
# API: Lambdas, integrations, routes and permissions
# ---------------------------------------------------------------------------

moved {
  from = aws_apigatewayv2_api.http
  to   = module.api.aws_apigatewayv2_api.this
}

moved {
  from = aws_apigatewayv2_stage.default
  to   = module.api.aws_apigatewayv2_stage.default
}

moved {
  from = aws_iam_role.api_sentiment_lambda
  to   = module.api.module.sentiment.aws_iam_role.this
}

moved {
  from = aws_iam_role_policy_attachment.api_sentiment_lambda_basic
  to   = module.api.module.sentiment.aws_iam_role_policy_attachment.basic
}

moved {
  from = aws_iam_role_policy.api_sentiment_lambda
  to   = module.api.module.sentiment.aws_iam_role_policy.this[0]
}

moved {
  from = aws_lambda_function.api_sentiment
  to   = module.api.module.sentiment.aws_lambda_function.this
}

moved {
  from = aws_apigatewayv2_integration.api_sentiment
  to   = module.api.aws_apigatewayv2_integration.sentiment
}

moved {
  from = aws_apigatewayv2_route.post_sentiment_by_symbol
  to   = module.api.aws_apigatewayv2_route.post_sentiment_by_symbol
}

moved {
  from = aws_lambda_permission.apigw_invoke_api_sentiment
  to   = module.api.aws_lambda_permission.sentiment
}

moved {
  from = aws_iam_role.api_cache_read_lambda
  to   = module.api.module.cache_read.aws_iam_role.this
}

moved {
  from = aws_iam_role_policy_attachment.api_cache_read_lambda_basic
  to   = module.api.module.cache_read.aws_iam_role_policy_attachment.basic
}

moved {
  from = aws_iam_role_policy.api_cache_read_lambda
  to   = module.api.module.cache_read.aws_iam_role_policy.this[0]
}

moved {
  from = aws_lambda_function.api_cache_read
  to   = module.api.module.cache_read.aws_lambda_function.this
}

moved {
  from = aws_apigatewayv2_integration.api_cache_read
  to   = module.api.aws_apigatewayv2_integration.cache_read
}

moved {
  from = aws_apigatewayv2_route.get_sentiment_cache_all
  to   = module.api.aws_apigatewayv2_route.get_sentiment_cache_all
}

moved {
  from = aws_apigatewayv2_route.get_sentiment_cache
  to   = module.api.aws_apigatewayv2_route.get_sentiment_cache
}

moved {
  from = aws_lambda_permission.apigw_invoke_api_cache_read
  to   = module.api.aws_lambda_permission.cache_read
}

moved {
  from = aws_iam_role.api_ticker_suggest_lambda
  to   = module.api.module.ticker_suggest.aws_iam_role.this
}

moved {
  from = aws_iam_role_policy_attachment.api_ticker_suggest_lambda_basic
  to   = module.api.module.ticker_suggest.aws_iam_role_policy_attachment.basic
}

# Already counted in the old configuration; a no-op when valid_tickers_ssm_param is unset.
moved {
  from = aws_iam_role_policy.api_ticker_suggest_lambda[0]
  to   = module.api.module.ticker_suggest.aws_iam_role_policy.this[0]
}

moved {
  from = aws_lambda_function.api_ticker_suggest
  to   = module.api.module.ticker_suggest.aws_lambda_function.this
}

moved {
  from = aws_apigatewayv2_integration.api_ticker_suggest
  to   = module.api.aws_apigatewayv2_integration.ticker_suggest
}

moved {
  from = aws_apigatewayv2_route.get_ticker_suggestions
  to   = module.api.aws_apigatewayv2_route.get_ticker_suggestions
}

moved {
  from = aws_lambda_permission.apigw_invoke_api_ticker_suggest
  to   = module.api.aws_lambda_permission.ticker_suggest
}

# ---------------------------------------------------------------------------
# SageMaker: IAM, registry, promotion and the retrain schedule
# ---------------------------------------------------------------------------

moved {
  from = aws_iam_role.sagemaker_execution
  to   = module.sagemaker.aws_iam_role.execution
}

moved {
  from = aws_iam_role_policy.sagemaker_inline
  to   = module.sagemaker.aws_iam_role_policy.execution
}

moved {
  from = aws_iam_role.sagemaker_pipeline
  to   = module.sagemaker.aws_iam_role.pipeline
}

moved {
  from = aws_iam_role_policy.sagemaker_pipeline_inline
  to   = module.sagemaker.aws_iam_role_policy.pipeline
}

moved {
  from = aws_sagemaker_model_package_group.sentiment
  to   = module.sagemaker.aws_sagemaker_model_package_group.this
}

moved {
  from = aws_iam_role.model_promote_lambda
  to   = module.sagemaker.module.model_promote.aws_iam_role.this
}

moved {
  from = aws_iam_role_policy_attachment.model_promote_lambda_basic
  to   = module.sagemaker.module.model_promote.aws_iam_role_policy_attachment.basic
}

moved {
  from = aws_iam_role_policy.model_promote_lambda
  to   = module.sagemaker.module.model_promote.aws_iam_role_policy.this[0]
}

moved {
  from = aws_lambda_function.model_promote
  to   = module.sagemaker.module.model_promote.aws_lambda_function.this
}

moved {
  from = aws_cloudwatch_event_rule.model_approved
  to   = module.sagemaker.aws_cloudwatch_event_rule.model_approved
}

moved {
  from = aws_cloudwatch_event_target.model_approved
  to   = module.sagemaker.aws_cloudwatch_event_target.model_approved
}

moved {
  from = aws_lambda_permission.eventbridge_invoke_model_promote
  to   = module.sagemaker.aws_lambda_permission.model_promote_events
}

moved {
  from = aws_iam_role.retrain_invoke
  to   = module.sagemaker.aws_iam_role.retrain_invoke
}

moved {
  from = aws_iam_role_policy.retrain_invoke
  to   = module.sagemaker.aws_iam_role_policy.retrain_invoke
}

moved {
  from = aws_cloudwatch_event_rule.retrain
  to   = module.sagemaker.aws_cloudwatch_event_rule.retrain
}

moved {
  from = aws_cloudwatch_event_target.retrain
  to   = module.sagemaker.aws_cloudwatch_event_target.retrain
}

# ---------------------------------------------------------------------------
# GitHub Actions OIDC
# ---------------------------------------------------------------------------

moved {
  from = aws_iam_openid_connect_provider.github
  to   = module.github_oidc.aws_iam_openid_connect_provider.github
}

moved {
  from = aws_iam_role.github_plan
  to   = module.github_oidc.aws_iam_role.plan
}

moved {
  from = aws_iam_role_policy_attachment.github_plan_readonly
  to   = module.github_oidc.aws_iam_role_policy_attachment.plan_readonly
}

moved {
  from = aws_iam_role_policy.github_plan_state
  to   = module.github_oidc.aws_iam_role_policy.plan_state
}

moved {
  from = aws_iam_role.github_apply
  to   = module.github_oidc.aws_iam_role.apply
}

moved {
  from = aws_iam_role_policy_attachment.github_apply_poweruser
  to   = module.github_oidc.aws_iam_role_policy_attachment.apply_poweruser
}

moved {
  from = aws_iam_role_policy.github_apply_state
  to   = module.github_oidc.aws_iam_role_policy.apply_state
}

moved {
  from = aws_iam_role_policy.github_apply_iam
  to   = module.github_oidc.aws_iam_role_policy.apply_iam
}

# ---------------------------------------------------------------------------
# Pre-existing removal, carried over unchanged.
#
# The Financial PhraseBank corpus is seeded by hand (see README); Terraform used to
# upload it, but `filemd5()` reads the file on every plan and CI has no copy of a
# licence-gated corpus. `destroy = false` makes Terraform forget the object instead of
# deleting it from S3. Safe to delete this block once it has been applied on main.
# ---------------------------------------------------------------------------

removed {
  from = aws_s3_object.phrasebank

  lifecycle {
    destroy = false
  }
}
