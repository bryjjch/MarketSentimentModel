data "archive_file" "finsense_shared_layer" {
  type        = "zip"
  source_dir  = abspath("${path.module}/../lambda/_layer")
  output_path = "${path.module}/build/finsense_shared_layer.zip"
  excludes    = ["__pycache__", "*.pyc", ".pytest_cache"]
}

resource "aws_lambda_layer_version" "finsense_shared" {
  layer_name          = "${var.project_name}-finsense-shared"
  filename            = data.archive_file.finsense_shared_layer.output_path
  source_code_hash    = data.archive_file.finsense_shared_layer.output_base64sha256
  compatible_runtimes = ["python3.12"]
  description         = "Shared helpers for Finsense ingestion/prediction/pseudo-label Lambdas"
}

data "archive_file" "finsense_deps_layer" {
  type        = "zip"
  source_dir  = abspath("${path.module}/../lambda/_deps_layer")
  output_path = "${path.module}/build/finsense_deps_layer.zip"
  excludes    = ["__pycache__", "*.pyc", ".pytest_cache"]
}

resource "aws_lambda_layer_version" "finsense_deps" {
  layer_name          = "${var.project_name}-finsense-deps"
  filename            = data.archive_file.finsense_deps_layer.output_path
  source_code_hash    = data.archive_file.finsense_deps_layer.output_base64sha256
  compatible_runtimes = ["python3.12"]
  description         = "Third-party Python SDK dependencies for Finsense Lambdas"
}
