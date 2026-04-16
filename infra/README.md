# AWS inference slice (S3, SageMaker, API Gateway)

This folder implements the "backwards-first" path: package a locally trained Hugging Face classifier, upload it to **S3**, deploy a **SageMaker Serverless Inference** endpoint with the **Hugging Face inference DLC**, and front it with an **HTTP API** (API Gateway v2) plus a **Lambda** that calls `InvokeEndpoint`.

## Layout

| Path | Purpose |
|------|---------|
| [`sagemaker/serving/code/inference.py`](sagemaker/serving/code/inference.py) | SageMaker `model_fn` / `input_fn` / `predict_fn` / `output_fn`; mirrors train/serve behavior from `training.inference` (max length 175, empty-text handling). |
| [`scripts/package_model_tarball.py`](scripts/package_model_tarball.py) | Builds `model.tar.gz` (HF weights at tarball root + `code/inference.py`; skips `checkpoint-*` dirs). |
| [`lambda/predict/handler.py`](lambda/predict/handler.py) | Lambda proxy: forwards JSON body to SageMaker runtime. |
| [`terraform/`](terraform/) | Declarative AWS resources (S3, IAM, SageMaker model/endpoint, Lambda, HTTP API). |

## 1. Package the model

From the repository root (with your trained folder, e.g. `outputs/clf_finbert`):

```bash
python infra/scripts/package_model_tarball.py --model_dir outputs/clf_finbert --output model.tar.gz
```

Use the **final** saved weights under `output_dir` (the script skips `checkpoint-*` directories). The tarball must include `code/inference.py` for the Hugging Face container.

## 2. SageMaker container image (DLC)

Pick a **Hugging Face PyTorch inference** DLC URI for the **same region** you will deploy. AWS publishes images per region (ECR account `763104351234` in many commercial regions, but always confirm in the current [AWS Deep Learning Containers](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html) / SageMaker documentation).

Choose a tag that is compatible with your stack (this repo pins inference libraries in [`requirements/pinned-serve.txt`](../requirements/pinned-serve.txt): PyTorch 2.5.x, Transformers 5.5.x). If the closest DLC uses a slightly older Transformers, test the endpoint before production.

Set `sagemaker_image_uri` in `terraform.tfvars` (see [`terraform/terraform.tfvars.example`](terraform/terraform.tfvars.example)).

## 3. S3 bucket behavior (Terraform)

The stack creates (or uses) a private model bucket with:

- **Block Public Access** on all four settings.
- **Bucket owner enforced** object ownership.
- **SSE-S3 (AES256)** default encryption.
- **Versioning** enabled (easy rollback when overwriting the same key).
- **Bucket policy** denying requests where `aws:SecureTransport` is false (TLS-only).

The SageMaker execution role is granted **`s3:GetObject` / `s3:ListBucket`** on that bucket (and **`logs:*`** under `/aws/sagemaker/*`, plus **ECR read** for pulling the DLC). No separate bucket policy is required for SageMaker to read objects; IAM on the execution role is sufficient.

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

**Order of creation:** S3 upload (`aws_s3_object`) → SageMaker model → endpoint configuration → endpoint → Lambda IAM + function → API routes. The first `apply` can take **15–25+ minutes** while the endpoint becomes `InService`.

## 5. SageMaker endpoint (Terraform)

- **Execution role**: trust `sagemaker.amazonaws.com`; inline policy for S3 model read, CloudWatch Logs under `/aws/sagemaker/*`, and ECR pulls for the DLC.
- **Model**: `PrimaryContainer` with your `sagemaker_image_uri` and `model_data_url` pointing at `s3://.../models/finsense/v1/model.tar.gz` (prefix configurable via `model_key_prefix`).
- **Endpoint**: one production variant using `serverless_config`; tune memory and concurrency with `sagemaker_serverless_memory_size_in_mb` and `sagemaker_serverless_max_concurrency`.

After deploy, outputs include `sagemaker_endpoint_name` and `predict_url`.

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

### Auth options for a future web app

- **JWT authorizer** (Amazon Cognito user pools) on the HTTP API.
- **Lambda authorizer** for API keys or custom tokens.
- **Do not** leave anonymous public access on production; pair throttling and WAF as needed.

## 7. Updating the model

1. Re-run training; package a new `model.tar.gz`.
2. `terraform apply` (updated `etag` on `aws_s3_object` triggers replacement where needed). SageMaker may require a **new model version** and **endpoint update**; Terraform replaces dependent resources when the object or model resource changes. For advanced blue/green deployments, extend the Terraform or use SageMaker deployment operations separately.

## 8. Cost notes

The largest ongoing cost is SageMaker inference usage. Serverless avoids paying for idle provisioned instances, but you should still delete the stack when not needed (`terraform destroy`; ensure `s3_force_destroy` if you need the bucket emptied).
