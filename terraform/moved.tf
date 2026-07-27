# State migration for inlining the s3-bucket module into modules/storage.
#
# Both buckets used to be instances of a shared ../s3-bucket module; they are now
# declared directly in models.tf and data.tf. Every resource kept its AWS identity —
# only its Terraform address changed — so these blocks rewrite state instead of
# destroying and recreating two buckets that hold model artifacts and pipeline data.
#
# Safe to delete once applied against every environment holding state (prod is the only
# one today). Until then, removing a block would orphan the resource it names.

# ---------------------------------------------------------------------------
# Model artifact bucket
# ---------------------------------------------------------------------------

moved {
  from = module.storage.module.models.aws_s3_bucket.this
  to   = module.storage.aws_s3_bucket.models
}

moved {
  from = module.storage.module.models.aws_s3_bucket_public_access_block.this
  to   = module.storage.aws_s3_bucket_public_access_block.models
}

moved {
  from = module.storage.module.models.aws_s3_bucket_ownership_controls.this
  to   = module.storage.aws_s3_bucket_ownership_controls.models
}

moved {
  from = module.storage.module.models.aws_s3_bucket_server_side_encryption_configuration.this
  to   = module.storage.aws_s3_bucket_server_side_encryption_configuration.models
}

moved {
  from = module.storage.module.models.aws_s3_bucket_versioning.this
  to   = module.storage.aws_s3_bucket_versioning.models
}

moved {
  from = module.storage.module.models.aws_s3_bucket_policy.tls
  to   = module.storage.aws_s3_bucket_policy.models_tls
}

# ---------------------------------------------------------------------------
# Pipeline data bucket
#
# The lifecycle configuration already lived in modules/storage, so it does not move.
# ---------------------------------------------------------------------------

moved {
  from = module.storage.module.data.aws_s3_bucket.this
  to   = module.storage.aws_s3_bucket.data
}

moved {
  from = module.storage.module.data.aws_s3_bucket_public_access_block.this
  to   = module.storage.aws_s3_bucket_public_access_block.data
}

moved {
  from = module.storage.module.data.aws_s3_bucket_ownership_controls.this
  to   = module.storage.aws_s3_bucket_ownership_controls.data
}

moved {
  from = module.storage.module.data.aws_s3_bucket_server_side_encryption_configuration.this
  to   = module.storage.aws_s3_bucket_server_side_encryption_configuration.data
}

moved {
  from = module.storage.module.data.aws_s3_bucket_versioning.this
  to   = module.storage.aws_s3_bucket_versioning.data
}

moved {
  from = module.storage.module.data.aws_s3_bucket_policy.tls
  to   = module.storage.aws_s3_bucket_policy.data_tls
}
