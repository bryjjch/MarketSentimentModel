# FinSense infrastructure

Terraform for the whole AWS stack. The root module here is what CI runs
(`working-directory: terraform`); everything it manages lives in a child module under
`modules/`.

## Layout

| Path | Purpose |
|------|---------|
| `main.tf` | The only place modules are wired together — read this first |
| `locals.tf` | Derived names, the constructed endpoint ARN, and the Lambda image list |
| `variables.tf` | All root inputs, bannered by the module that consumes them |
| `outputs.tf` | Read by name from `scripts/` and `.github/workflows/` — rename with care |
| `moved.tf` | State migration for the s3-bucket inlining; see the note in the file |
| `env/prod.tfvars` | Production values, applied by `.github/workflows/deploy.yml` |
| `backend.hcl` | S3 backend config, passed to `terraform init -backend-config=` |

## Modules

`lambda-function` is a shared primitive used by the others; the rest each own one slice
of the stack.

| Module | Kind | Owns |
|--------|------|------|
| `lambda-function` | primitive | A container-image Lambda plus its role, basic execution policy and inline policy |
| `ecr` | component | One immutable-tag repository per Lambda image |
| `storage` | component | Model + data buckets, data lifecycle rules, sentiment cache table |
| `queues` | component | The four stage-to-stage SQS queues and their DLQs |
| `pipeline` | component | Daily cron -> dispatch -> collect -> predict -> label -> cache-write |
| `api` | component | HTTP API, its three Lambdas, routes and invoke permissions |
| `sagemaker` | component | SageMaker IAM, model package group, model promotion, retrain schedule |
| `github-oidc` | component | CI plan/apply roles (bootstrap only — apply once from a workstation) |

Adding a Lambda means adding a key to `local.lambda_images` in `locals.tf` (which
creates its ECR repository) and calling `modules/lambda-function` from the module that
owns it.

## Usage

```bash
terraform init -backend-config=backend.hcl
terraform plan  -var-file=env/prod.tfvars
```

`image_tag` is required. Locally it comes from `image_tag.auto.tfvars`, written by
`scripts/lambda-build-push.sh`; CI passes `-var` from `scripts/image-tag.sh`.

## What Terraform deliberately does not manage

- **The SageMaker model, endpoint config and endpoint.** The `model_promote` Lambda
  creates and rolls them forward on each model-package approval. See the model
  registry note in `modules/sagemaker/main.tf`.
- **The SageMaker Pipeline definition.** Generated and upserted by
  `src/sagemaker/pipeline/build_pipeline.py` in CI, after Terraform has run.
- **The Financial PhraseBank corpus** in `s3://<data bucket>/reference/phrasebank/`,
  seeded by hand.
