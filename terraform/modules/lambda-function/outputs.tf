output "function_name" {
  description = "Lambda function name."
  value       = aws_lambda_function.this.function_name
}

output "arn" {
  description = "Lambda function ARN."
  value       = aws_lambda_function.this.arn
}

output "invoke_arn" {
  description = "ARN used as an API Gateway AWS_PROXY integration_uri."
  value       = aws_lambda_function.this.invoke_arn
}
