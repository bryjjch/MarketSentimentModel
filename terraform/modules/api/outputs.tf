output "invoke_url" {
  description = "Base URL for the HTTP API's $default stage."
  value       = aws_apigatewayv2_stage.default.invoke_url
}
