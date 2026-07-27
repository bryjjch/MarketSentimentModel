# FinSense infrastructure

Terraform for the whole AWS stack. The root module here is what CI runs
(`working-directory: terraform`); everything it manages lives in a child module under
`modules/`.

## Setup

### Root files

| Path | Purpose |
|------|---------|
| `main.tf` | The only place modules are wired together — read this first |
| `locals.tf` | Derived names, the constructed endpoint ARN, and the Lambda image list |
| `variables.tf` | All root inputs, bannered by the module that consumes them |
| `outputs.tf` | Read by name from `scripts/` and `.github/workflows/` — rename with care |
| `versions.tf` | Terraform ≥ 1.5, AWS provider ~> 5.0, and the S3 backend declaration |
| `providers.tf` | The AWS provider and the caller-identity lookup |
| `moved.tf` | State migration for the s3-bucket inlining; see the note in the file |
| `env/prod.tfvars` | Production values, applied by `.github/workflows/deploy.yml` |
| `backend.hcl` | S3 backend config, passed to `terraform init -backend-config=` |
| `terraform.tfvars.example` | Template for local-only overrides (secret ARNs, dev toggles) |

`backend.hcl` and `.terraform.lock.hcl` are deliberately committed: CI works from a fresh
clone, so `init -backend-config=backend.hcl` needs the file present, and an untracked lock
would let CI resolve a different provider version than the one planned locally. The state
bucket and lock table it names are created out of band — Terraform cannot create its own
backend.

### Running it

```bash
terraform init -backend-config=backend.hcl
terraform plan  -var-file=env/prod.tfvars
terraform apply -var-file=env/prod.tfvars
```

`terraform.tfvars` is loaded automatically on top of `env/prod.tfvars`, so for a
workstation apply it only needs the values you want to differ from production. Copy it
from `terraform.tfvars.example` and put your Secrets Manager ARNs there; they are absent
from `env/prod.tfvars` because they carry the account ID, and CI supplies them as `TF_VAR_*`
environment variables instead.

`image_tag` is required and has no default. Locally it comes from
`image_tag.auto.tfvars`, written by `scripts/lambda-build-push.sh` after it pushes images;
CI passes `-var image_tag=...` from `scripts/image-tag.sh`.

`modules/github-oidc` is the one bootstrap: apply it once from a workstation with admin
credentials to create the OIDC provider and the CI roles, then every apply after that runs
in CI. If the account already has a GitHub OIDC provider, import it rather than creating a
second one — AWS permits only one per URL.

## Architecture

Modules are wired only in `main.tf`, so the dependency graph reads top to bottom:

```
ecr ─────────► image URIs, consumed by every module that declares a Lambda
storage ─────► bucket names/ARNs + the cache table ──┐
queues ──────► queue URLs/ARNs ──────────────────────┤
                                                     ├──► pipeline
locals.endpoint_name / endpoint_arn ─────────────────┤
                                                     └──► api
sagemaker ───► SageMaker IAM, registry, promotion, retrain schedule
github_oidc ─► CI plan/apply roles (bootstrap only)
```

`lambda-function` is a shared primitive used by the others; the rest each own one slice of
the stack.

| Module | Kind | Owns |
|--------|------|------|
| `lambda-function` | primitive | A container-image Lambda plus its role, basic execution policy and inline policy |
| `ecr` | component | One immutable-tag repository per Lambda image |
| `storage` | component | Model + data buckets, data lifecycle rules, sentiment cache table |
| `queues` | component | The four stage-to-stage SQS queues and their DLQs |
| `pipeline` | component | Daily cron → dispatch → collect → predict → label → cache-write |
| `api` | component | HTTP API, its three Lambdas, routes and invoke permissions |
| `sagemaker` | component | SageMaker IAM, model package group, model promotion, retrain schedule |
| `github-oidc` | component | CI plan/apply roles (bootstrap only — apply once from a workstation) |

### `lambda-function` (primitive)

Every FinSense Lambda has the same shape — an IAM role, `AWSLambdaBasicExecutionRole`, one
inline policy, one image-backed function — so callers describe only what differs. Names are
derived from `var.name` (`<name>` for the function, `<name>-lambda` for the role,
`<name>-perms` for the policy). A function needing nothing but CloudWatch Logs passes
`policy_json = null`.

This is why the component modules read as flat lists of stages rather than dozens of IAM
resources.

### `ecr`

One repository per entry in `local.lambda_images`, with immutable tags. The names are
derived from `project_name`, matching what `scripts/lambda-build-push.sh` and the CI matrix
push to. `locals.tf` turns each repository URL plus `var.image_tag` into the `image_uris`
map that every other module indexes.

Adding a Lambda means adding a key to `local.lambda_images` — which creates its repository
— and calling `modules/lambda-function` from the module that owns it.

### `storage`

The two S3 buckets and the DynamoDB table. The buckets share a baseline (private,
encrypted, versioned, TLS-only) but are declared separately rather than through a shared
wrapper: they are the only two the project has and they already diverge, since only the
data bucket carries lifecycle rules. The cost is that a baseline change has to be made
twice.

Exports bucket names and ARNs plus the cache table name and ARN, which `pipeline` and
`api` scope their IAM policies against.

### `queues`

Four queues, each with a dead-letter queue, driven from one `var.queues` map — the key is
what callers index `queue_urls` / `queue_arns` by. Visibility timeouts are declared here
next to the queue at 6× the consumer Lambda's timeout (AWS guidance for event source
mappings) rather than left to a default, and retention is set per queue: four days for the
pipeline stages, one day for cache writes, because stale sentiment is worthless.

### `pipeline`

The daily ingestion chain: the SSM ticker parameter, the EventBridge cron rule, five
Lambdas via the `lambda-function` primitive, and one event source mapping per queue
consumer. Stages are declared in execution order, with shared values in `locals`.

It is the module with the most inputs, because every tuning knob of the data flywheel —
schedule, article counts, confidence thresholds, LLM provider and model, per-function
memory — lands here.

### `api`

The HTTP API, its stage with CORS and throttling, and three Lambdas each declared right
next to the integration, route and invoke permission that expose it. It shares the
cache-write queue with `pipeline`, since `api_sentiment` writes back through the same
`cache_write` consumer.

### `sagemaker`

Two IAM roles — the inference execution role assumed by the endpoint, and the pipeline
role assumed by training and processing steps — plus the model package group, the
`model_promote` Lambda, the EventBridge rule that fires it on approval, and the retrain
schedule (disabled by default, with its own role for `StartPipelineExecution`).

Note what is *missing*: there is no `aws_sagemaker_model`, `endpoint_configuration`, or
`endpoint` resource. Terraform owns the endpoint's name, execution role and serverless
sizing — passed to the promoter as environment variables — and `model_promote` owns the
resources themselves.

### `github-oidc`

The OIDC provider and two roles, with the trust policy's `sub` pinned to this repository —
without which any repo on GitHub could assume them. The plan role gets read-only plus the
state lock; the apply role gets `PowerUserAccess` plus a narrow inline IAM policy scoped to
the project's naming prefix, and can only be assumed by a run on the default branch.
