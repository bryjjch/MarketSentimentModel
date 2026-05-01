# Infrastructure (S3, SageMaker, API Gateway)

This folder implements the AWS infrastructure for the Finsense pipeline. Training runs on **SageMaker** and writes a deploy-ready artifact directly (weights + tokenizer + `code/inference.py`) to `SM_MODEL_DIR`, which SageMaker auto-packs into `model.tar.gz` at job end. Terraform wires that tarball into a **SageMaker Serverless Inference** endpoint with the **Hugging Face inference DLC**, fronted by an **HTTP API** (API Gateway v2) plus a **Lambda** that calls `InvokeEndpoint`.

A second **daily ingestion pipeline** (EventBridge → ingestion Lambda → ingestion-prediction Lambda → pseudo-label Lambda) writes raw news/social text, model predictions, low-confidence pseudo-labels, and curated training rows to a **separate data bucket** partitioned by `dt=YYYY-MM-DD/symbol=SYM/…`. See [§7](#7-daily-ingestion--provider-agnostic-pseudo-labeling).

## Layout

| Path | Purpose |
|------|---------|
| [`sagemaker/serving/code/inference.py`](sagemaker/serving/code/inference.py) | SageMaker `model_fn` / `input_fn` / `predict_fn` / `output_fn`; mirrors train/serve behavior from `training.inference` (max length 175, empty-text handling). Copied into `<output_dir>/code/` by the classifier training script so the SageMaker-produced `model.tar.gz` is serving-ready. |
| [`lambda/api_inference/handler.py`](lambda/api_inference/handler.py) | Lambda proxy: forwards JSON body to SageMaker runtime (raw text only). |
| [`lambda/api_sentiment_by_symbol/`](lambda/api_sentiment_by_symbol/) | Orchestration: resolve symbol, collect news (**Finnhub** symbol-keyed company news when `finnhub_secret_arn` is set; else Google News RSS) and optional Reddit posts, batch `invoke_endpoint`, aggregate scores. |
| [`lambda/cache_read/handler.py`](lambda/cache_read/handler.py) | Reads precomputed per-symbol rows from DynamoDB (`GET /sentiment/cache/{symbol}`). |
| [`lambda/ingestion/handler.py`](lambda/ingestion/handler.py) | **Daily ingestion** Lambda: EventBridge fan-out that collects raw news/social text per ticker and writes it to `raw/` in the data bucket, then async-invokes the prediction Lambda per symbol. |
| [`lambda/ingestion_prediction/handler.py`](lambda/ingestion_prediction/handler.py) | Reads one `raw/` partition, runs SageMaker in batches, writes per-text rows to `predictions/`, writes high-confidence rows to `curated/`, fans low-confidence rows out to the pseudo-label Lambda, and refreshes the DynamoDB `sentiment_cache` row used by the existing read API. |
| [`lambda/pseudo_label/handler.py`](lambda/pseudo_label/handler.py) | Provider-agnostic LLM labeler (`openai`, `google`, or offline `echo`). Writes `pseudo/` rows and merges newly-labeled rows into `curated/` with `source=pseudo`. |
| [`lambda/_layer/python/finsense_shared/`](lambda/_layer/python/finsense_shared/) | Shared Lambda code layer: source adapters, S3 I/O, SageMaker batch invoker, confidence math, Hive-style key helpers, and LLM labeling logic. |
| [`lambda/_deps_layer/`](lambda/_deps_layer/) | Shared Lambda dependency layer (`python/` site-packages) for third-party SDKs such as OpenAI and Google GenAI. |
| [`terraform/`](terraform/) | Declarative AWS resources (S3, IAM, SageMaker model/endpoint, Lambda, HTTP API, DynamoDB, EventBridge). |

## 1. Obtain `model.tar.gz`

`finsense-train-classifier` is SageMaker-native: when launched as a training job it writes weights, tokenizer, `training_manifest.json`, and `code/inference.py` to `SM_MODEL_DIR`, and SageMaker automatically uploads the packaged `model.tar.gz` to the job's `OutputDataConfig.S3OutputPath`. Download that object (or point `model_tarball_path` at an `s3://` copy you control) before running `terraform apply`.

For local iteration you can still run training on your laptop (`finsense-train-classifier --output_dir outputs/clf_finbert`): the same layout is produced under `outputs/clf_finbert/` and you only need to `tar -czf model.tar.gz -C outputs/clf_finbert .` to hand it to Terraform.

## 2. SageMaker container image (DLC)

Pick a **Hugging Face PyTorch inference** DLC URI for the **same region** you will deploy. Confirm available DLCs [`here`](https://huggingface.co/docs/sagemaker/dlcs/available).

Set `sagemaker_image_uri` in `terraform.tfvars` (see [`terraform/terraform.tfvars.example`](terraform/terraform.tfvars.example)).

## 3. S3 bucket behavior (Terraform)

The stack creates **two** private buckets. Both share the same security baseline:

- **Block Public Access** on all four settings.
- **Bucket owner enforced** object ownership.
- **SSE-S3 (AES256)** default encryption.
- **Versioning** enabled (easy rollback when overwriting the same key).
- **Bucket policy** denying requests where `aws:SecureTransport` is false (TLS-only).

1. **Models bucket** (`${project_name}-models-${account_id}` unless `bucket_name` overrides) holds the serving `model.tar.gz`. The SageMaker execution role is granted **`s3:GetObject` / `s3:ListBucket`** on that bucket (and **`logs:*`** under `/aws/sagemaker/*`, plus **ECR read** for pulling the DLC).
2. **Data bucket** (`${project_name}-data-${account_id}` unless `data_bucket_name` overrides) holds the daily ingestion pipeline output (`raw/`, `predictions/`, `pseudo/`, `curated/`). Lifecycle rules expire `raw/` and `predictions/` partitions after `var.data_retention_days` days (default 90); `pseudo/` and `curated/` are retained indefinitely because they double as training data.

Ingestion / prediction / pseudo-label Lambdas each get **prefix-scoped** S3 permissions (e.g. the ingestion Lambda can only write to `raw/*`, the prediction Lambda can only write to `predictions/*` and `curated/*`). See §7.

For development destroys, you may set `s3_force_destroy = true` in `terraform.tfvars` so `terraform destroy` can empty both buckets (use with care).

## 4. Deploy with Terraform

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: aws_region, sagemaker_image_uri, model_tarball_path

# One-time remote state setup:
cp backend.hcl.example backend.hcl
# Edit backend.hcl: bucket, key, region, dynamodb_table

# First time with remote state (or when switching from local state):
terraform init -backend-config=backend.hcl -reconfigure -migrate-state
terraform apply
```

To bootstrap the backend storage itself, create the S3 bucket and lock table once (outside this stack), then keep reusing them:

```bash
aws s3api create-bucket --bucket <state-bucket-name> --region us-east-1
aws s3api put-bucket-versioning --bucket <state-bucket-name> --versioning-configuration Status=Enabled
aws dynamodb create-table \
  --table-name <lock-table-name> \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

After bootstrapping, day-to-day commands from any computer are:

```bash
cd infra/terraform
terraform init -backend-config=backend.hcl
terraform plan
terraform apply
```

`model_tarball_path` is resolved with `abspath()` relative to your current working directory when you run `apply`; paths like `../../model.tar.gz` from `infra/terraform` work if `model.tar.gz` lives at the repo root.

**Order of creation:** S3 upload (`aws_s3_object`) → SageMaker model → endpoint configuration → endpoint → Lambda IAM + functions (api_inference, api_sentiment_by_symbol, cache_read, pseudo_label, ingestion_prediction, ingestion) → Lambda Layers (`finsense_shared` + `finsense_deps`) → API routes → DynamoDB / SSM → EventBridge ingestion schedule. The first `apply` can take **15–25+ minutes** while the endpoint becomes `InService`.

Before `terraform apply`, install dependency packages into the deps layer payload:

```bash
pip install --target infra/lambda/_deps_layer/python -r infra/lambda/_deps_layer/requirements.txt
```

## 5. SageMaker endpoint (Terraform)

- **Execution role**: trust `sagemaker.amazonaws.com`; inline policy for S3 model read, CloudWatch Logs under `/aws/sagemaker/*`, and ECR pulls for the DLC.
- **Model**: `PrimaryContainer` with your `sagemaker_image_uri` and `model_data_url` pointing at `s3://.../models/finsense/v1/model.tar.gz` (prefix configurable via `model_key_prefix`).
- **Endpoint**: one production variant using `serverless_config`; tune memory and concurrency with `sagemaker_serverless_memory_size_in_mb` and `sagemaker_serverless_max_concurrency`.

After deploy, outputs include `sagemaker_endpoint_name`, `predict_url`, `sentiment_by_symbol_url`, `sentiment_cache_read_url_template`, `sentiment_cache_table_name`, `data_bucket_name`, `ingestion_function_name`, `ingestion_prediction_function_name`, `pseudo_label_function_name`, and `ingestion_rule_name`.

## 6. API Gateway + Lambda

- **HTTP API** (v2) with **CORS** (`cors_allow_origins`, default `["*"]` for prototyping; restrict in production).
- **Stage throttling**: `default_route_settings` on `$default` with `apigateway_throttle_rate_limit` (RPS) and `apigateway_throttle_burst_limit` (burst).
- **Route** `POST /predict` → **Lambda** (Python 3.12) with `boto3` `invoke_endpoint`, `ContentType` / `Accept` `application/json`.
- **Lambda reserved concurrency**: `lambda_reserved_concurrent_executions` caps parallel invokes (default `5`; align with `sagemaker_serverless_max_concurrency`). Use `-1` for no reservation.
- **Lambda IAM**: `sagemaker:InvokeEndpoint` scoped to the created endpoint ARN; plus `AWSLambdaBasicExecutionRole` for CloudWatch Logs.

### Example request

```bash
curl -sS -X POST "$(terraform output -raw predict_url)" \
  -H "Content-Type: application/json" \
  -d '{"text":"EPS beat expectations"}'
```

Body formats supported by the SageMaker handler match [`inference.py`](sagemaker/serving/code/inference.py): `{"text":"..."}` or `{"texts":["...","..."]}`. The Lambda forwards the body as-is.

### Sentiment by symbol (`POST /sentiment/by-symbol`)

A separate **orchestration Lambda** (`lambda/api_sentiment_by_symbol/`) implements `POST /sentiment/by-symbol` on the same HTTP API. It:

1. Normalizes the ticker (uppercase, 1–5 letters).
2. Collects text via **source adapters**: Google News RSS (no API key) and, if `options.include_social` is true (default), **Reddit** via the official API when credentials are configured.
3. Calls **SageMaker** with `{"texts":[...]}` (same contract as [`inference.py`](sagemaker/serving/code/inference.py)) using `boto3` `invoke_endpoint`—not the `POST /predict` route.
4. Aggregates per-text probabilities into `score` (mean of positive minus negative probability mass per article), `label` (`positive` / `neutral` / `negative`), `article_count`, and `recent_headlines` (title + URL pairs; length capped by `RECENT_HEADLINES_MAX`, default 10).

**Request (example):**

```bash
curl -sS -X POST "$(terraform output -raw sentiment_by_symbol_url)" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","options":{"max_articles":12,"include_social":true}}'
```

**Reddit (optional):** Create a Secrets Manager secret with JSON `{"client_id":"...","client_secret":"..."}` and set `reddit_credentials_secret_arn` in `terraform.tfvars`. If unset or empty, Reddit is skipped and only news RSS is used.

**API Gateway timeout:** HTTP API integrations are limited to about **30 seconds**. Keep `max_articles` modest so RSS fetch + SageMaker stay within that window; if you need heavier scraping, move work to an async pattern (e.g. SQS + worker) or raise caps only in offline jobs.

### DynamoDB cache

Terraform provisions:

- Table **`{project_name}-sentiment-cache`**: partition key `symbol` (string); attributes include `score`, `label`, `article_count`, `recent_headlines`, `updated_at`, `expires_at`. **TTL** is enabled on `expires_at` (write time + `sentiment_cache_ttl_seconds`, default 7 days).
- **SSM** parameter `/{project_name}/top-tickers`: JSON array of tickers consumed by the daily ingestion Lambda (editable in AWS Console or via Terraform `top_tickers_json`).

The cache row is written by the **ingestion-prediction Lambda** as part of the daily ingestion fan-out (see [§7](#7-daily-ingestion--provider-agnostic-pseudo-labeling)) — the older dedicated `sentiment_refresh` Lambda has been removed since its work is now a side-effect of the predict path.

### Read cached snapshot (`GET /sentiment/cache/{symbol}`)

```bash
curl -sS "$(terraform output -raw http_api_invoke_url)sentiment/cache/AAPL"
```

Returns the DynamoDB item as JSON, or `404` if missing. Clients can prefer this for hot symbols to avoid repeating full orchestration and SageMaker calls.

### Auth options for a future web app

- **JWT authorizer** (Amazon Cognito user pools) on the HTTP API.
- **Lambda authorizer** for API keys or custom tokens.
- **Do not** leave anonymous public access on production; pair throttling and WAF as needed.

## 7. Daily ingestion & provider-agnostic pseudo-labeling

A second, independent pipeline feeds the training corpus by running every day at
`var.ingestion_schedule` (default `cron(0 13 * * ? *)` = 13:00 UTC). It stores everything
in a **separate S3 bucket** (`var.data_bucket_name`, defaults to
`${project_name}-data-${account_id}`) so the model artifact bucket stays single-purpose.

### Data layout

The pipeline writes Hive-partitioned JSON Lines:

```
s3://<data-bucket>/
  raw/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl         # titles/URLs/snippets from RSS+Reddit
  predictions/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl # rows + probabilities + confidence
  pseudo/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl      # LLM labels for low-confidence rows
  curated/dt=YYYY-MM-DD/symbol=AAPL/<run_id>.jsonl         # high-conf model labels (source=model)
  curated/dt=YYYY-MM-DD/symbol=AAPL/<run_id>-pseudo.jsonl  # LLM labels (source=pseudo)
```

Athena / Glue / SageMaker Processing can register the bucket as a single table and
filter by `dt` / `symbol` / `source` (model vs pseudo) without reshuffling data.

### Flow

```
EventBridge (daily)
    └── ingestion Lambda
         ├── writes raw/dt=.../symbol=.../
         └── async-invoke (per symbol) ──► ingestion-prediction Lambda
                                                ├── InvokeEndpoint (SageMaker)
                                                ├── writes predictions/…
                                                ├── writes curated/… (high-conf)
                                                ├── PutItem sentiment_cache (DynamoDB)
                                                └── async-invoke ──► pseudo-label Lambda
                                                                          ├── calls OpenAI / Gemini
                                                                          ├── writes pseudo/…
                                                                          └── writes curated/…-pseudo
```

### Confidence gating

A prediction is treated as **low-confidence** and routed to the pseudo-label Lambda when
the model's top-class probability is below `var.low_conf_top_prob` (default `0.65`), or
optionally when the margin between the top two classes is below `var.low_conf_margin`
(default `0.0`, disabled). Tune both in `terraform.tfvars` without touching code.

### Pseudo-labeler provider selection

`var.llm_provider` selects the backend:

- `openai` (default): Chat Completions, model `var.llm_model` (default `gpt-4o-mini`).
- `google`: Gemini via Google AI Studio; default model `gemini-2.0-flash`.
- `echo`: deterministic offline stub (plumbing tests only, not real labels).

API keys are sourced from Secrets Manager when `var.openai_secret_arn` /
`var.google_secret_arn` are set (JSON shape `{"api_key":"..."}`), otherwise from the
runtime env vars `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `GEMINI_API_KEY`.

### On-demand vs scheduled overlap

The original `sentiment_refresh` Lambda did `collect → predict → cache-write` per
ticker and the `POST /sentiment/by-symbol` Lambda did the same thing on demand for API
Gateway traffic. The daily ingestion pipeline supersedes the scheduled refresh: the
ingestion-prediction Lambda writes the same DynamoDB `sentiment_cache` rows the UI already reads
via `GET /sentiment/cache/{symbol}`, so no consumer changes are required.

The `sentiment_refresh` Lambda, its IAM role, and its EventBridge rule have been
**removed** from this stack. The on-demand `POST /sentiment/by-symbol` Lambda stays
because it serves interactive traffic with a different access pattern. Existing
deployments will see `terraform apply` destroy the legacy resources cleanly.

### Manual replay / single-symbol run

```bash
aws lambda invoke --function-name "$(terraform output -raw ingestion_function_name)" \
  --payload '{"symbol":"AAPL","options":{"max_articles":10}}' /tmp/ingest.json
```

## 8. SageMaker training pipeline

A **SageMaker Pipeline** orchestrates the full training workflow: data preparation,
MLM continued pre-training, classifier fine-tuning, held-out evaluation, quality gating,
and model registration.

### Pipeline DAG

```
DataPrep (Processing)
    -> MLM Pre-Training (Training)
    -> Classifier Training (Training)
    -> Evaluate Classifier (Processing)
    -> Check macro_f1 >= threshold (Condition)
        [pass] -> Register Model (Model Package)
        [fail] -> pipeline ends without registration
```

### Pipeline code layout

| Path | Purpose |
|------|---------|
| [`sagemaker/pipeline/pipeline_definition.py`](sagemaker/pipeline/pipeline_definition.py) | Builds the `sagemaker.workflow.pipeline.Pipeline` object via the Python SDK. |
| [`sagemaker/pipeline/build_pipeline.py`](sagemaker/pipeline/build_pipeline.py) | CLI helper to generate pipeline definition JSON (or upsert directly). |
| [`sagemaker/pipeline/scripts/prepare_training_data.py`](sagemaker/pipeline/scripts/prepare_training_data.py) | Processing script: assembles curated data + PhraseBank (from S3 `reference/phrasebank/`), produces MLM corpus, classifier data, and held-out test split. |
| [`sagemaker/pipeline/scripts/evaluate_classifier.py`](sagemaker/pipeline/scripts/evaluate_classifier.py) | Processing script: loads trained classifier, evaluates against the held-out test set, writes `evaluation.json`. |
| [`sagemaker/pipeline/entry_points/run_mlm.py`](sagemaker/pipeline/entry_points/run_mlm.py) | Thin wrapper for `training.train_mlm:main` (resolves relative imports under SageMaker). |
| [`sagemaker/pipeline/entry_points/run_classifier.py`](sagemaker/pipeline/entry_points/run_classifier.py) | Thin wrapper for `training.train_classifier:main`. |

### Build and deploy the pipeline

1. Install pipeline build dependencies (SageMaker Python SDK **v2**; v3 is incompatible with this code path):

```bash
cd <repo-root>
pip install -r requirements/pinned-pipeline.txt
```

2. Generate the pipeline definition JSON from the Python SDK:

```bash
cd <repo-root>
python -m infra.sagemaker.pipeline.build_pipeline \
    --role arn:aws:iam::123456789012:role/finsense-sagemaker-pipeline \
    --region us-east-1 \
    --output infra/terraform/pipeline_definition.json
```

3. Deploy with Terraform:

```bash
cd infra/terraform
terraform apply   # picks up pipeline_definition.json automatically
```

4. Start a pipeline execution (AWS CLI):

```bash
aws sagemaker start-pipeline-execution \
    --pipeline-name "$(terraform output -raw pipeline_name)" \
    --pipeline-parameters '[
        {"Name":"DataBucket","Value":"finsense-data-123456789012"},
        {"Name":"CuratedS3Prefix","Value":"s3://finsense-data-123456789012/curated/"}
    ]'
```

### Pipeline parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DataBucket` | session default | S3 bucket for intermediate pipeline artifacts. |
| `CuratedS3Prefix` | `curated/` | S3 prefix for curated training data from the ingestion pipeline. |
| `PhraseBankS3Prefix` | `reference/phrasebank/` | S3 prefix where `Sentences_75Agree.txt` is stored (uploaded by Terraform). |
| `BaseModel` | `bert-base-uncased` | Hugging Face checkpoint for MLM + classifier. |
| `TrainImageUri` | HF PyTorch training DLC | Training container image. |
| `InferenceImageUri` | HF PyTorch inference DLC | Inference image used when registering the model package. |
| `ProcessingInstanceType` | `ml.m5.xlarge` | Instance type for data prep and evaluation. |
| `TrainingInstanceType` | `ml.g4dn.xlarge` | Instance type for MLM and classifier training. |
| `MlmEpochs` | `3` | MLM pre-training epochs. |
| `ClfEpochs` | `3` | Classifier fine-tuning epochs. |
| `ModelPackageGroup` | `finsense-sentiment` | Model Package Group for accepted models. |
| `MacroF1Threshold` | `0.80` | Minimum macro F1 on the held-out test set to register. |
| `TestRatio` | `0.10` | Fraction of PhraseBank held out for evaluation. |
| `Seed` | `42` | Random seed for reproducibility. |

### Terraform variables (pipeline-specific)

| Variable | Default | Description |
|----------|---------|-------------|
| `pipeline_name` | `{project_name}-training-pipeline` | SageMaker Pipeline resource name. |
| `pipeline_definition_json` | `""` | Inline JSON (takes precedence over file). |
| `pipeline_definition_path` | `pipeline_definition.json` | Path to the generated JSON file. |
| `model_package_group_name` | `finsense-sentiment` | Model Package Group. |
| `phrasebank_path` | `../../data/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_75Agree.txt` | Local path to PhraseBank file, uploaded to the data bucket under `reference/phrasebank/`. |
| `pipeline_macro_f1_threshold` | `0.80` | Threshold (for documentation; runtime value is a pipeline parameter). |

## 9. Updating the model

1. Launch a new SageMaker training job (or re-run locally and repackage `outputs/<run>/` into a fresh `model.tar.gz`); point `model_tarball_path` at the new artifact.
2. `terraform apply` (updated `etag` on `aws_s3_object` triggers replacement where needed). SageMaker may require a **new model version** and **endpoint update**; Terraform replaces dependent resources when the object or model resource changes. For advanced blue/green deployments, extend the Terraform or use SageMaker deployment operations separately.
3. **Pipeline-driven updates**: run a pipeline execution instead of manual training. When the model passes the quality gate, it is registered in the Model Package Group. Promote the latest approved version by updating `model_tarball_path` to point at the registered artifact and running `terraform apply`.

## 10. Cost notes

The largest ongoing cost is SageMaker inference usage. Serverless avoids paying for idle provisioned instances, but you should still delete the stack when not needed (`terraform destroy`; ensure `s3_force_destroy` if you need the bucket emptied).
