# FinSense source

Three packages, split by where the code runs.

| Folder | Runs in | Purpose |
|--------|---------|---------|
| `lambdas/` | AWS Lambda (container images) | Everything that runs per request or per pipeline task: the daily ingestion chain, the HTTP API handlers, the cache writer, and the model promoter — plus `finsense_shared`, the code layer baked into all nine images |
| `sagemaker/` | SageMaker (pipeline steps and the endpoint) | The training pipeline's DAG definition and step scripts, and the inference handler that ships inside `model.tar.gz` |
| `training/` | SageMaker training jobs, or locally | The actual modeling code: data loading, MLM pre-training, classifier fine-tuning, evaluation, LLM pseudo-labeling |

## How they relate

`lambdas/` and `training/` never import each other. They meet at two contracts:

- **The S3 data layout.** The Lambdas write `curated/` partitions; the training pipeline's
  data-prep step reads them. Both sides agree on the Hive-partitioned JSON Lines shape
  defined in `lambdas/finsense_shared/pipeline.py`.
- **The prediction record.** `training/inference.py` and `sagemaker/serving/code/inference.py`
  produce the same per-text record — `label_id`, `label_name`, `probabilities` keyed by
  sentiment name — and `lambdas/finsense_shared/sentiment.py` is written against it. The
  serving copy deliberately duplicates the logic rather than importing `training`, because
  it ships inside the model artifact.

`sagemaker/` is the bridge: its entry points wrap `training/`'s CLIs, and the training
package is copied into each step's source tarball by the estimator's `dependencies`.

Each folder has its own README with the details.
