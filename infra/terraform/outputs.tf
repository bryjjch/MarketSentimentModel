output "s3_bucket_name" {
  description = "Bucket storing model.tar.gz"
  value       = aws_s3_bucket.models.bucket
}

output "s3_model_key" {
  description = "S3 object key for the uploaded model artifact"
  value       = aws_s3_object.model_artifact.key
}

output "sagemaker_endpoint_name" {
  description = "Name passed to InvokeEndpoint / Lambda env SAGEMAKER_ENDPOINT_NAME"
  value       = aws_sagemaker_endpoint.classifier.name
}

output "sagemaker_endpoint_arn" {
  value = aws_sagemaker_endpoint.classifier.arn
}

output "http_api_invoke_url" {
  description = "Base URL for the HTTP API (POST {url}/predict)"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "predict_url" {
  description = "Full URL for POST /predict"
  value       = "${trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")}/predict"
}

output "sentiment_by_symbol_url" {
  description = "Full URL for POST /sentiment/by-symbol (JSON body: symbol, optional options)"
  value       = "${trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")}/sentiment/by-symbol"
}

output "sentiment_cache_read_url_template" {
  description = "GET cached snapshot: append symbol (e.g. .../sentiment/cache/AAPL)"
  value       = "${trimsuffix(aws_apigatewayv2_stage.default.invoke_url, "/")}/sentiment/cache/{symbol}"
}

output "sentiment_cache_table_name" {
  description = "DynamoDB table for precomputed per-symbol sentiment"
  value       = aws_dynamodb_table.sentiment_cache.name
}

output "sentiment_refresh_rule_name" {
  description = "EventBridge rule name for scheduled cache refresh"
  value       = aws_cloudwatch_event_rule.sentiment_refresh.name
}
