# FinSense

FinSense is a financial sentiment analysis stack built around a fine-tuned FinBERT classifier. It covers model training, AWS deployment, and a React UI for per-symbol sentiment heatmaps.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/training/` | Training package: data helpers, pseudo-labeling, MLM, classifier, inference, metrics, run manifests |
| `terraform/` | AWS resources, split into modules for storage, queues, the ingestion pipeline, the HTTP API, SageMaker and CI OIDC roles (see `terraform/README.md`) |
| `.github/workflows/` | `deploy.yml` (plan on PR, apply on merge) and `train.yml` (manual training / promotion) |
| `src/sagemaker/` | SageMaker serving handler and training pipeline (build script, processing scripts, training entry points) |
| `src/lambdas/` | Lambda handlers and shared `finsense_shared` code layer |
| `scripts/` | Local build helpers and the one-time `migrate-state-to-ci.sh` |
| `frontend/` | React + TypeScript + Vite UI for cache list / heatmap |
| `notebooks/` | Exploratory / demonstration notebooks |
| `data/` | Local download location for Financial PhraseBank, used by the notebooks (created on first use). Cloud training reads the copy in S3 instead |
| `tests/` | Pytest suite |

## Environment setup

All Python dependencies live in `pyproject.toml`. The base project has no dependencies;
each environment is an extra, so you install only what you need:

```bash
python -m venv .venv && source .venv/bin/activate

pip install -e ".[train,dev]"           # training, pseudo-labeling, and the test suite
pip install -e ".[train,dev,pipeline]"  # the above plus the SageMaker SDK — what CI runs
pip install -e ".[pipeline]"            # SDK only, no torch: builds/upserts the pipeline
pip install -e ".[serve]"               # inference-only stack
```

Run the suite with `pytest`. The pipeline-definition tests skip without the `pipeline`
extra, so use the CI combination if you are changing `pipeline_definition.py`.

Versions are pinned exactly — these describe reproducible environments, not a library's
compatible range. Note that neither cloud training nor the deployed endpoint installs
them: both run in AWS Deep Learning Containers pinned by `TrainImageUri` /
`InferenceImageUri` in `src/sagemaker/pipeline/pipeline_definition.py`.

## Architecture decisions

### Two-stage training: MLM → classifier

The pipeline first runs **masked language model (MLM) pre-training** on the domain corpus. This adapts the model's representations to financial news language before the classification head is added. The SageMaker training pipeline enforces this order: DataPrep → MLM → Classifier → Evaluate → quality gate → Register.

### SageMaker Serverless Inference

The model is served via a **Serverless Inference** endpoint rather than a provisioned instance. This eliminates idle costs for our workload with infrequent demand. The tradeoff is cold-start latency on the first invocation after a quiet period.

### Two API paths: on-demand vs. pre-computed cache

Two access patterns co-exist on the same HTTP API:

- **`POST /sentiment/by-symbol`** — collects live news, calls the SageMaker endpoint, and returns a fresh score. Suited for interactive queries.
- **`GET /sentiment/cache/{symbol}`** — reads a pre-computed DynamoDB row with a 7-day TTL. Suited for the UI heatmap, which reads the same symbols repeatedly.

Both paths feed the same DynamoDB table through one `cache_write` Lambda: the daily pipeline and the on-demand API each enqueue a cache-write task on SQS, and `cache_write` is the only function with write access to the table.

### Daily data flywheel with confidence-gated pseudo-labeling

An EventBridge-triggered chain of single-purpose Lambdas, connected by SQS queues (each with a dead-letter queue), runs once per day:

1. **Dispatch** (`pipeline_dispatch`) — enumerates tickers, mints a shared `run_id`, and enqueues one collect task per symbol.
2. **Collect** (`pipeline_collect`) — collects news and social text for one ticker, writes `raw/` to S3.
3. **Predict** (`pipeline_predict`) — scores each text, writes `predictions/`. High-confidence rows go directly to `curated/` as training data; the aggregated score is enqueued for `cache_write`.
4. **Label** (`pipeline_label`) — low-confidence rows (top-class probability < 0.65 by default) are routed to an LLM (OpenAI or Gemini, provider-agnostic). The LLM label is written to `pseudo/` and merged into `curated/`.

This loop continuously expands the labeled training corpus without manual annotation.

### Deployment is a push, not a checklist

Pushing to `main` runs `.github/workflows/deploy.yml`, which builds and pushes the nine Lambda images, applies Terraform, and upserts the SageMaker Pipeline. Pull requests get a `terraform plan` posted as a comment. CI authenticates with GitHub OIDC (`terraform/modules/github-oidc/`) — there are no AWS keys anywhere.

Three things deliberately stay outside Terraform:

- **The training pipeline.** Compiling its definition uploads `sourcedir.tar.gz` to S3 and embeds the role ARN Terraform creates, so it can only be built *after* an apply. CI runs `build_pipeline.py --upsert` as a post-apply step.
- **The SageMaker model, endpoint config and endpoint.** These are created by the `model_promote` Lambda. Describing them in Terraform would mean every deploy needed the 405 MB `model.tar.gz` on disk, and every retrain would surface as drift.
- **The Financial PhraseBank corpus.** It is seeded once into `s3://<data bucket>/reference/phrasebank/` and read from there by the pipeline's `PhraseBankS3Prefix` input. A Terraform-managed upload would call `filemd5()` on a local copy during *every* plan, which no CI runner has.

Seeding it is a one-off, from any machine that has the corpus:

```bash
aws s3 cp data/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_75Agree.txt \
  "s3://$(terraform -chdir=terraform output -raw data_bucket_name)/reference/phrasebank/"
```

Lambda images are tagged with the git tree hash of `src/lambdas` rather than the commit SHA, so a Terraform-only commit reuses the existing tag and rebuilds nothing.

### Retraining is gated on approval, not on a deploy

The training pipeline runs in the cloud — on a schedule (`retrain_schedule`, disabled by default) or via the `train` workflow. Nothing trains locally.

A run that clears the macro-F1 threshold registers a model package as **`PendingManualApproval`**. Approving it — in the SageMaker console, or `aws sagemaker update-model-package --model-approval-status Approved` — emits an EventBridge event that invokes `model_promote`, which mints a new model and endpoint config and updates the endpoint in place. A run that misses the threshold registers nothing.

Rollback is the same mechanism in reverse: re-approve an older package, or invoke `model_promote` with its ARN. The last `model_versions_to_keep` (default 3) generations are retained for exactly this.

### Hive-partitioned S3 layout

All pipeline output is written in Hive-style partitions (`dt=YYYY-MM-DD/symbol=AAPL/`) so the entire data bucket can be registered as a single Athena table and queried by date, symbol, or source (`model` vs. `pseudo`) without reshuffling data.

### Label convention

Training and saved models use integer labels aligned with Financial PhraseBank:

| Label | Sentiment |
|-------|-----------|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

FinBERT checkpoints ship with a different default class order; the classifier training script remaps weights so all saved artifacts use the table above.
