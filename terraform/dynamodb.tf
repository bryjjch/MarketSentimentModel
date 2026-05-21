resource "aws_dynamodb_table" "sentiment_cache" {
  name         = "${var.project_name}-sentiment-cache"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "symbol"

  attribute {
    name = "symbol"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

resource "aws_ssm_parameter" "top_tickers" {
  name  = "/${var.project_name}/top-tickers"
  type  = "String"
  value = var.top_tickers_json

  description = "JSON array of tickers for sentiment cache refresher (edit in console or terraform)."
}
