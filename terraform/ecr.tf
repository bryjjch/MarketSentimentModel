resource "aws_ecr_repository" "pipeline_dispatch" {
  name                 = "${var.project_name}-pipeline-dispatch"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "pipeline_collect" {
  name                 = "${var.project_name}-pipeline-collect"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "pipeline_predict" {
  name                 = "${var.project_name}-pipeline-predict"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "pipeline_label" {
  name                 = "${var.project_name}-pipeline-label"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "cache_write" {
  name                 = "${var.project_name}-cache-write"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "api_sentiment" {
  name                 = "${var.project_name}-api-sentiment"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "api_cache_read" {
  name                 = "${var.project_name}-api-cache-read"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "api_ticker_suggest" {
  name                 = "${var.project_name}-api-ticker-suggest"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}
