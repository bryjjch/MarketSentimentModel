# Infrastructure (S3, SageMaker, API Gateway)

This folder implements the AWS infrastructure for the Finsense pipeline. Currently, it follows a "backwards-first" path: package a locally trained Hugging Face classifier, upload it to **S3**, deploy a **SageMaker Serverless Inference** endpoint with the **Hugging Face inference DLC**, and front it with an **HTTP API** (API Gateway v2) plus a **Lambda** that calls `InvokeEndpoint`.

## Layout

| Path | Purpose |
|------|---------|
| [`sagemaker/serving/code/inference.py`](sagemaker/serving/code/inference.py) | SageMaker `model_fn` / `input_fn` / `predict_fn` / `output_fn`; mirrors train/serve behavior from `training.inference` (max length 175, empty-text handling). |
| [`scripts/package_model_tarball.py`](scripts/package_model_tarball.py) | Builds `model.tar.gz` (HF weights at tarball root + `code/inference.py`; skips `checkpoint-*` dirs). |
| [`lambda/predict/handler.py`](lambda/predict/handler.py) | Lambda proxy: forwards JSON body to SageMaker runtime (raw text only). |
| [`lambda/sentiment/`](lambda/sentiment/) | Orchestration: resolve symbol, collect news (RSS) and optional Reddit posts, batch `invoke_endpoint`, aggregate scores. |
| [`lambda/cache_read/handler.py`](lambda/cache_read/handler.py) | Reads precomputed per-symbol rows from DynamoDB (`GET /sentiment/cache/{symbol}`). |
| [`lambda/sentiment_refresh/handler.py`](lambda/sentiment_refresh/handler.py) | Scheduled job: invokes sentiment Lambda per ticker, writes DynamoDB cache. |
| [`terraform/`](terraform/) | Declarative AWS resources (S3, IAM, SageMaker model/endpoint, Lambda, HTTP API, DynamoDB, EventBridge). |

## 1. Package the model

From the repository root (with your trained folder, e.g. `outputs/clf_finbert`):

```bash
python infra/scripts/package_model_tarball.py --model_dir outputs/clf_finbert --output model.tar.gz
```

Use the **final** saved weights under `output_dir` (the script skips `checkpoint-*` directories). The tarball must include `code/inference.py` for the Hugging Face container.

## 2. SageMaker container image (DLC)

Pick a **Hugging Face PyTorch inference** DLC URI for the **same region** you will deploy. Confirm available DLCs [`here`](https://huggingface.co/docs/sagemaker/dlcs/available).

Set `sagemaker_image_uri` in `terraform.tfvars` (see [`terraform/terraform.tfvars.example`](terraform/terraform.tfvars.example)).

## 3. S3 bucket behavior (Terraform)

The stack creates (or uses) a private model bucket with:

- **Block Public Access** on all four settings.
- **Bucket owner enforced** object ownership.
- **SSE-S3 (AES256)** default encryption.
- **Versioning** enabled (easy rollback when overwriting the same key).
- **Bucket policy** denying requests where `aws:SecureTransport` is false (TLS-only).

The SageMaker execution role is granted **`s3:GetObject` / `s3:ListBucket`** on that bucket (and **`logs:*`** under `/aws/sagemaker/*`, plus **ECR read** for pulling the DLC).

For development destroys, you may set `s3_force_destroy = true` in `terraform.tfvars` so `terraform destroy` can empty the bucket (use with care).

## 4. Deploy with Terraform

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: aws_region, sagemaker_image_uri, model_tarball_path

terraform init
terraform apply
```

`model_tarball_path` is resolved with `abspath()` relative to your current working directory when you run `apply`; paths like `../../model.tar.gz` from `infra/terraform` work if `model.tar.gz` lives at the repo root.

**Order of creation:** S3 upload (`aws_s3_object`) → SageMaker model → endpoint configuration → endpoint → Lambda IAM + functions → API routes → DynamoDB / SSM / scheduled refresh. The first `apply` can take **15–25+ minutes** while the endpoint becomes `InService`.

## 5. SageMaker endpoint (Terraform)

- **Execution role**: trust `sagemaker.amazonaws.com`; inline policy for S3 model read, CloudWatch Logs under `/aws/sagemaker/*`, and ECR pulls for the DLC.
- **Model**: `PrimaryContainer` with your `sagemaker_image_uri` and `model_data_url` pointing at `s3://.../models/finsense/v1/model.tar.gz` (prefix configurable via `model_key_prefix`).
- **Endpoint**: one production variant using `serverless_config`; tune memory and concurrency with `sagemaker_serverless_memory_size_in_mb` and `sagemaker_serverless_max_concurrency`.

After deploy, outputs include `sagemaker_endpoint_name`, `predict_url`, `sentiment_by_symbol_url`, `sentiment_cache_read_url_template`, `sentiment_cache_table_name`, and `sentiment_refresh_rule_name`.

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

A separate **orchestration Lambda** (`lambda/sentiment/`) implements `POST /sentiment/by-symbol` on the same HTTP API. It:

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

### DynamoDB cache and scheduled refresh

Terraform provisions:

- Table **`{project_name}-sentiment-cache`**: partition key `symbol` (string); attributes written by the refresher include `score`, `label`, `article_count`, `recent_headlines`, `updated_at`, `expires_at`. **TTL** is enabled on `expires_at` (refresh time + `sentiment_cache_ttl_seconds`, default 7 days).
- **SSM** parameter `/{project_name}/top-tickers`: JSON array of tickers for the refresher (editable in AWS Console or via Terraform `top_tickers_json`).
- **EventBridge** rule (`sentiment_refresh_schedule`, default `rate(1 hour)`) invoking **`{project_name}-sentiment-refresh`**, which **invokes the sentiment Lambda directly** (same logic as the HTTP route) once per ticker and `PutItem`s into DynamoDB.

### Read cached snapshot (`GET /sentiment/cache/{symbol}`)

```bash
curl -sS "$(terraform output -raw http_api_invoke_url)sentiment/cache/AAPL"
```

Returns the DynamoDB item as JSON, or `404` if missing. Clients can prefer this for hot symbols to avoid repeating full orchestration and SageMaker calls.

### Auth options for a future web app

- **JWT authorizer** (Amazon Cognito user pools) on the HTTP API.
- **Lambda authorizer** for API keys or custom tokens.
- **Do not** leave anonymous public access on production; pair throttling and WAF as needed.

## 7. Updating the model

1. Re-run training; package a new `model.tar.gz`.
2. `terraform apply` (updated `etag` on `aws_s3_object` triggers replacement where needed). SageMaker may require a **new model version** and **endpoint update**; Terraform replaces dependent resources when the object or model resource changes. For advanced blue/green deployments, extend the Terraform or use SageMaker deployment operations separately.

## 8. Cost notes

The largest ongoing cost is SageMaker inference usage. Serverless avoids paying for idle provisioned instances, but you should still delete the stack when not needed (`terraform destroy`; ensure `s3_force_destroy` if you need the bucket emptied).
