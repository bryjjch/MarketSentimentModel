output "dispatch_function_name" {
  description = "Dispatch Lambda (EventBridge cron target); enqueues one collect task per ticker."
  value       = module.dispatch.function_name
}

output "dispatch_rule_name" {
  description = "EventBridge rule name for the daily pipeline dispatch."
  value       = aws_cloudwatch_event_rule.dispatch.name
}

output "top_tickers_parameter_name" {
  description = "SSM parameter holding the ticker list the dispatch Lambda fans out over."
  value       = aws_ssm_parameter.top_tickers.name
}
