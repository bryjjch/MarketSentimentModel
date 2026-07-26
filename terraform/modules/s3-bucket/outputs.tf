output "id" {
  description = "Bucket id (same as its name)."
  value       = aws_s3_bucket.this.id
}

output "bucket" {
  description = "Bucket name."
  value       = aws_s3_bucket.this.bucket
}

output "arn" {
  description = "Bucket ARN."
  value       = aws_s3_bucket.this.arn
}
