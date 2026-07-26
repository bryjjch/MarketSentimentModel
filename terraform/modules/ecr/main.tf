# One immutable-tag repository per Lambda image. scripts/lambda-build-push.sh derives
# these names from project_name, so the keys here are the image names CI pushes.

resource "aws_ecr_repository" "this" {
  for_each = var.repositories

  name                 = "${var.project_name}-${each.key}"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}
