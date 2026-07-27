# Everything in the project that holds state:
#
#   models.tf  -> s3://<models bucket>, written by SageMaker training jobs
#   data.tf    -> s3://<data bucket>, the pipeline's raw/predictions/pseudo/curated
#   cache.tf   -> the DynamoDB sentiment cache the API reads
#
# The two buckets share a baseline (private, encrypted, versioned, TLS-only) and are
# deliberately declared separately rather than through a shared bucket module. They are
# the only two buckets the project has and they already diverge — only the data bucket
# carries lifecycle rules — so one file per bucket reads end to end without a hop
# through a wrapper. The cost is that a change to the baseline has to be made twice.
