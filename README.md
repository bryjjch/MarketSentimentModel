# FinSense

FinSense is a financial news sentiment tool: a fine-tuned FinBERT classifier scores news about a given equity as negative, neutral, or positive, surfaced as a per-symbol heatmap dashboard.

<img width="1484" height="766" alt="Screenshot 2026-07-28 at 9 31 42 PM" src="https://github.com/user-attachments/assets/acedfbab-e783-4d37-8759-d9f0557554f2" />

A daily job collects news and social text per ticker and scores it. Confident predictions are kept as new training data; uncertain ones are routed to an LLM for labeling -- so the labeled corpus grows every day with no manual annotation. A SageMaker training pipeline turns that corpus into a new model version, registered if it clears a macro-F1 gate and promoted to the live endpoint after human approval.

The whole system is serverless and reproducible: the model runs on a SageMaker Serverless Inference endpoint behind an HTTP API, and all infrastructure -- nine Lambdas, the API, buckets, queues, and IAM -- is defined in Terraform. Every push to main deploys.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/` | All application and model code — see `src/README.md` |
| `src/lambdas/` | The nine Lambda handlers and the `finsense_shared` code layer they share |
| `src/sagemaker/` | The training pipeline definition and the endpoint's serving handler |
| `src/training/` | The training package: data prep, MLM, classifier, evaluation, pseudo-labeling |
| `terraform/` | The whole AWS stack, split into modules — see `terraform/README.md` |
| `terraform/modules/` | One module per slice of the stack (storage, queues, pipeline, api, sagemaker, ecr, github-oidc) |
| `terraform/env/` | `prod.tfvars`, the values CI applies |
| `.github/workflows/` | `deploy.yml` — the only workflow: test, build images, apply, upsert the pipeline |
| `scripts/` | Local build helpers: `lambda-build-push.sh`, `image-tag.sh`, `check_lambda_sources.py` |
| `frontend/` | React + TypeScript + Vite dashboard (sentiment heatmap, ticker search) |
| `tests/` | Pytest suite covering the training package, the Lambda handlers, and the pipeline definition |
| `notebooks/` | Exploratory notebooks |
| `data/` | Local download location for Financial PhraseBank, used by the notebooks. Cloud training reads the copy in S3 instead |

## Environment setup

### Tools

| Tool | Needed for |
|------|-----------|
| Python ≥ 3.10 (CI uses 3.12) | The training package, the Lambda handlers, the pipeline build script |
| Docker | Building the Lambda container images (`linux/amd64`) |
| Terraform ≥ 1.5 (CI pins 1.15.8) | The infrastructure |
| AWS CLI v2, with credentials | Terraform, ECR pushes, reading outputs |
| Node ≥ 20 | The frontend |

### Python

The project has no base dependencies. Every environment is an extra in
`pyproject.toml`, so you install only what you need:

```bash
python -m venv .venv && source .venv/bin/activate

pip install -e ".[train,dev]"           # training, pseudo-labeling, and the test suite
pip install -e ".[train,dev,pipeline]"  # the above plus the SageMaker SDK — what CI runs
pip install -e ".[pipeline]"            # SDK only, no torch: builds/upserts the pipeline
pip install -e ".[serve]"               # inference-only stack
```

Run the suite with `pytest`. The pipeline-definition tests skip without the `pipeline`
extra, so use the CI combination if you are changing `src/sagemaker/pipeline/`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # set VITE_API_BASE_URL to the deployed API base URL
npm run dev
```

The API base URL comes from `terraform -chdir=terraform output -raw http_api_invoke_url`.
Add `http://localhost:5173` to `cors_allow_origins` for local development against a
deployed API.

### AWS, one time per account

The state backend, the CI roles, and a few pieces of data live outside the normal deploy
path. Do these once, from a workstation with admin credentials.

1. **Create the state backend** — the S3 bucket and DynamoDB lock table named in
   `terraform/backend.hcl` (`finsense-terraform-state-bucket`,
   `finsense-terraform-states-table`). Terraform cannot create its own backend.

2. **Store the API keys** in Secrets Manager, each as JSON `{"api_key": "..."}`:
   a Google AI Studio key (required — `pipeline_label` cannot label without it), an
   OpenAI key (optional, alternative provider), and optionally Reddit credentials as
   `{"client_id": "...", "client_secret": "..."}` to enable the social source.

3. **Apply once locally** to create everything, including the GitHub OIDC provider and the
   CI roles. This takes three commands, because the Lambdas cannot be created before their
   images exist and the ECR repositories that hold those images are themselves Terraform
   resources:

   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars   # put your secret ARNs here
   terraform init -backend-config=backend.hcl

   # 1. the registries only
   terraform apply -var-file=env/prod.tfvars -target=module.ecr \
     -var "image_tag=$(../scripts/image-tag.sh)"

   # 2. build and push the nine images (also writes image_tag.auto.tfvars)
   ../scripts/lambda-build-push.sh

   # 3. everything else
   terraform apply -var-file=env/prod.tfvars
   ```

   Only the first apply needs `-var image_tag=...`; after step 2 the tag is picked up
   automatically from the generated `image_tag.auto.tfvars`.

4. **Set the GitHub Actions repository variables** so CI can take over:

   | Variable | Value |
   |----------|-------|
   | `AWS_REGION` | e.g. `us-east-1` |
   | `AWS_PLAN_ROLE_ARN` | `terraform output -raw github_plan_role_arn` |
   | `AWS_APPLY_ROLE_ARN` | `terraform output -raw github_apply_role_arn` |
   | `TF_VAR_GOOGLE_SECRET_ARN` | the Google secret ARN (required) |
   | `TF_VAR_OPENAI_SECRET_ARN` | the OpenAI secret ARN (optional) |
   | `TF_VAR_REDDIT_CREDENTIALS_SECRET_ARN` | the Reddit secret ARN (optional) |

   Secret ARNs are not in `env/prod.tfvars` because they carry the account ID.

5. **Seed the Financial PhraseBank corpus.** The training pipeline reads it from S3, and
   it is licence-gated, so it is not in the repo and not a Terraform-managed object:

   ```bash
   aws s3 cp data/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_75Agree.txt \
     "s3://$(terraform -chdir=terraform output -raw data_bucket_name)/reference/phrasebank/"
   ```

After this, pushing to `main` is the only deploy step. There are no AWS keys in the
repository or in GitHub — CI authenticates with OIDC.

## AWS architecture

```
                 EventBridge (daily cron)
                          │
                          ▼
                 ┌─────────────────┐
                 │pipeline_dispatch│  one collect task per ticker (SSM ticker list)
                 └────────┬────────┘
                     [collect queue]
                          ▼
                 ┌─────────────────┐      s3://data/raw/dt=…/symbol=…/
                 │ pipeline_collect│─────────────────────────────────►
                 └────────┬────────┘
                     [predict queue]
                          ▼
                 ┌─────────────────┐◄────► SageMaker Serverless endpoint
                 │ pipeline_predict│─────► s3://data/predictions/ + curated/ (high conf)
                 └───┬─────────┬───┘
      [cache-write queue]   [label queue]
                 │             ▼
                 │      ┌──────────────┐        LLM (Gemini / OpenAI)
                 │      │pipeline_label│◄──────►
                 │      └──────┬───────┘
                 │             └───────► s3://data/pseudo/ + curated/ (LLM labels)
                 ▼
          ┌─────────────┐
          │ cache_write │──────► DynamoDB sentiment_cache
          └─────────────┘
                 ▲
      [cache-write queue]
                 │
   ┌─────────────┴───────────────────────────────────┐
   │ API Gateway HTTP API                            │
   │   POST /sentiment/by-symbol   → api_sentiment ──┘ (live collect + score)
   │   GET  /sentiment/cache[/{symbol}] → api_cache_read  (reads DynamoDB)
   │   GET  /tickers/suggest       → api_ticker_suggest
   └─────────────────────────────────────────────────┘
```

### Storage

Two S3 buckets and one DynamoDB table, all in `modules/storage`:

- **Models bucket** — model artifacts, written by SageMaker training jobs. Nothing is
  uploaded from a workstation.
- **Data bucket** — the pipeline's `raw/`, `predictions/`, `pseudo/` and `curated/`
  partitions, plus the seeded `reference/phrasebank/` corpus. Everything is written in
  Hive-style partitions (`dt=YYYY-MM-DD/symbol=AAPL/`) as JSON Lines, so the bucket can
  be registered as a single Athena table and queried by date, symbol, or label source. A
  lifecycle rule expires `raw/` and `predictions/` after `data_retention_days` (90);
  `pseudo/` and `curated/` are kept indefinitely because they are training data.
- **DynamoDB `sentiment_cache`** — one row per symbol with the latest score, label,
  article count and recent headlines. TTL cleanup runs off `expires_at`.

Both buckets are private, encrypted, versioned and TLS-only.

### Daily ingestion pipeline

An EventBridge cron fires `pipeline_dispatch` once a day (13:00 UTC). Each stage is a
single-purpose Lambda consuming one SQS queue and producing onto the next; the four
queues each have a dead-letter queue, and visibility timeouts are 6× the consumer's
timeout. No stage imports another — they agree only on the key layout and task shapes in
`finsense_shared.pipeline`, and every stage is idempotent, so redelivery is safe.

`pipeline_predict` is where the loop closes. Predictions whose top-class probability
clears `low_conf_top_prob` (0.65) go straight to `curated/` as training data; the rest
are routed to `pipeline_label`, which asks an LLM for a label and writes that to
`pseudo/` and `curated/`. The per-symbol aggregate goes onto the cache-write queue.

### HTTP API

An API Gateway HTTP API (v2) with CORS and stage-level throttling fronts three Lambdas.
Two access patterns coexist: `POST /sentiment/by-symbol` collects live text and invokes
the endpoint for a fresh score, while `GET /sentiment/cache/{symbol}` reads the
pre-computed DynamoDB row — which is what the UI heatmap uses, since it reads the same
symbols repeatedly.

Both paths reach DynamoDB through the same cache-write queue. `cache_write` is the only
function with write access to the table, and its put is conditional on `updated_at` not
regressing, so queue reordering can never overwrite a symbol with older data.

### Inference and retraining

The model is served from a **SageMaker Serverless Inference** endpoint, which removes
idle cost for a workload with infrequent demand at the price of cold-start latency after
a quiet period. Terraform owns the endpoint's name, execution role and serverless sizing
— but not the endpoint itself.

Retraining runs entirely in the cloud, either on the `retrain_schedule` (disabled by
default) or started by hand from the SageMaker console. A run that clears the macro-F1
threshold registers a model package as `PendingManualApproval`; a run that misses it
registers nothing. Approving the package emits an EventBridge event that invokes
`model_promote`, which mints a model and endpoint config and updates the endpoint in
place, pruning all but the last `model_versions_to_keep` (3) generations. Rollback is the
same mechanism in reverse: re-approve an older package.

## CI/CD

`.github/workflows/deploy.yml` is the only workflow. Pull requests get a plan; merging to
`main` deploys. Five jobs:

```
test ──┐
       ├──► images (matrix × 9) ──► terraform ──► sagemaker-pipeline
tag ───┘
```

1. **test** — installs `[train,dev,pipeline]`, checks every Lambda source compiles
   (`scripts/check_lambda_sources.py`), asserts the SageMaker SDK is importable so the
   pipeline-definition tests cannot silently skip, then runs `pytest`.
2. **tag** — computes the image tag once, so every downstream job agrees on it.
3. **images** — builds the nine Lambda images in parallel and pushes them to ECR. Pull
   requests build but never push; the plan role has no ECR write access.
4. **terraform** — `fmt -check`, `init`, `validate`, `plan` against `env/prod.tfvars`.
   On a pull request the plan is posted as a comment (one comment, updated in place); on
   `main` it is applied.
5. **sagemaker-pipeline** — runs `build_pipeline.py --upsert`, reading the role ARN,
   pipeline name and bucket from Terraform outputs.

A few details are load-bearing:

- **The image tag is the git tree hash of `src/lambdas`**, not the commit SHA
  (`scripts/image-tag.sh`). It changes if and only if Lambda source, dependencies or
  Dockerfiles change, so a Terraform-only or frontend-only commit reuses the existing
  tag, `terraform plan` shows nothing to do, and the ECR skip-check rebuilds nothing.
  Because a moving base image does not change the tree, forcing a rebuild after a base
  update needs the workflow's `rebuild_images` input.
- **ECR repositories use immutable tags**, so re-pushing an existing tag is a hard error.
  Each matrix job checks `describe-images` first and skips if the tag is already there.
- **CI assumes a role via GitHub OIDC** — the plan role on pull requests, the apply role
  on `main` — with the trust policy pinned to this repository and branch
  (`terraform/modules/github-oidc/`). There are no AWS keys anywhere.
- **Concurrency is serialized, not cancelled.** A queued run waits, so two applies never
  overlap; `init` and `plan` also use `-lock-timeout=5m` for contending pull requests.

Three things deliberately stay outside the Terraform apply:

- **The training pipeline definition.** Compiling it uploads `sourcedir.tar.gz` to S3 and
  embeds the role ARN Terraform creates, so it can only be built *after* an apply — hence
  the separate post-apply job. `upsert` is idempotent.
- **The SageMaker model, endpoint config and endpoint.** `model_promote` owns these.
  Describing them in Terraform would mean every deploy needed the 405 MB `model.tar.gz`
  on disk, and every retrain would surface as drift.
- **The Financial PhraseBank corpus.** A Terraform-managed upload would call `filemd5()`
  on a local copy during *every* plan, which no CI runner has.
