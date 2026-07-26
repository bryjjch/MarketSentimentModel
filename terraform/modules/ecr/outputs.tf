output "repository_urls" {
  description = "Repository suffix -> registry URL (no tag)."
  value       = { for key, repo in aws_ecr_repository.this : key => repo.repository_url }
}
