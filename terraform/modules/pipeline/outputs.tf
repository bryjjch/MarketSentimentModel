output "dispatch_function_name" {
  description = "Dispatch Lambda (EventBridge cron target); enqueues one collect task per ticker."
  value       = module.dispatch.function_name
}

output "dispatch_rule_name" {
  description = "EventBridge rule name for the daily pipeline dispatch."
  value       = aws_cloudwatch_event_rule.dispatch.name
}
