variable "project_name" {
  type        = string
  description = "Resource name prefix."
}

variable "repositories" {
  type        = set(string)
  description = "Repository suffixes; each becomes <project_name>-<suffix>."
}
