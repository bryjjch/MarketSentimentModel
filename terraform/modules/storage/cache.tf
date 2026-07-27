# ---------------------------------------------------------------------------
# DynamoDB: sentiment cache
#
# cache_write is the sole writer; the API reads it. TTL cleanup runs off expires_at.
# ---------------------------------------------------------------------------

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
