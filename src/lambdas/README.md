# Lambdas

Nine container-image Lambdas, each one directory with a `handler.py` and a `Dockerfile`,
plus `finsense_shared/` — the code layer copied into all nine images.

The functions never import each other. They communicate through SQS queues, S3 objects
and EventBridge events, and agree only on the key layout and task shapes defined in
`finsense_shared/pipeline.py`. That means adding or changing a stage touches one handler
and one contract, and every stage is idempotent (same `run_id` → same S3 keys; cache
writes are conditional), so SQS redelivery is always safe.

## The daily ingestion chain

```
EventBridge cron
      │
      ▼
pipeline_dispatch ──[collect queue]──► pipeline_collect ──[predict queue]──► pipeline_predict
                                             │                                    │
                                        raw/ in S3              predictions/ + curated/ in S3
                                                                                  │
                                        ┌─────────────────────────────────────────┴────┐
                                 [label queue]                            [cache-write queue]
                                        ▼                                              ▼
                                 pipeline_label ──► pseudo/ + curated/           cache_write ──► DynamoDB
```

### `pipeline_dispatch`

The cron entrypoint. Loads the ticker list (SSM `/{project}/top-tickers`), mints one
`run_id` shared by the whole run so every downstream partition lines up, and enqueues one
collect task per symbol. It does nothing else — no collection, no scoring.

Also accepts a direct invoke with `{"symbol": "AAPL"}` to run a single ticker on demand.

### `pipeline_collect`

Consumes the collect queue one message at a time. Gathers text for **one** symbol through
`finsense_shared.sources` (Google News RSS, then Reddit when credentials are configured),
drops empty rows, and writes the result to `raw/dt=…/symbol=…/<run_id>.jsonl`. If anything
was written, it enqueues a predict task pointing at that object.

The raw feed is kept durable so it can be replayed into training, and so later experiments
with different models or thresholds can reuse the same corpus.

### `pipeline_predict`

The hinge of the whole system. Reads one `raw/` object, invokes the SageMaker endpoint in
batches, and fans out four ways:

- writes every row with its probabilities to `predictions/dt=…/symbol=…/<run_id>.jsonl`;
- writes the **high-confidence** rows to `curated/…` with `source: "model"` — the model's
  own label is trusted, so this is immediately usable training data;
- enqueues a label task carrying only the **low-confidence** row *indices* (the label
  Lambda re-reads the rows itself, so the message stays tiny regardless of text length);
- enqueues the aggregated per-symbol score on the cache-write queue.

A row is low-confidence when its top-class probability falls below `LOW_CONF_TOP_PROB`
(0.65), or optionally when the margin to the runner-up is too small — see
`finsense_shared/sentiment.py`.

### `pipeline_label`

Consumes the label queue, re-reads the referenced rows out of the `predictions/` object by
`row_index`, and asks an LLM for a label. Writes both labels — the model's and the LLM's —
to `pseudo/…` so they can be compared, and writes the successfully labeled rows to
`curated/…/<run_id>-pseudo.jsonl` with `source: "pseudo"`, a sibling of the
high-confidence object `pipeline_predict` already wrote.

The provider is resolved at runtime (`google`, `openai`, or `echo` for offline plumbing
tests) with the key coming from Secrets Manager. A provider failure is logged per row and
does not kill the batch.

### `cache_write`

The sole owner of writes to the DynamoDB `sentiment_cache` table — no other function has
write access. Consumes batches of up to 10 and reports partial batch failures, so one bad
message does not replay the whole batch.

Its put is conditional on `updated_at` not regressing, so standard-queue reordering and
redeliveries can never overwrite a symbol with older data.

Two producers share its queue: `pipeline_predict` (the daily run) and `api_sentiment`
(on-demand requests).

## The HTTP API

### `api_sentiment` — `POST /sentiment/by-symbol`

The on-demand path. Validates the symbol against the ticker universe, collects live text
through the same `finsense_shared.sources` adapters `pipeline_collect` uses, invokes the
endpoint, aggregates, and returns the score synchronously. Its only side effect is a
best-effort cache-write task, so a symbol queried once is available to cache reads
afterwards.

SageMaker failures are mapped to HTTP status codes: model/invocation errors become 502,
unknown symbols 400.

### `api_cache_read` — `GET /sentiment/cache`, `GET /sentiment/cache/{symbol}`

A read-only view of the DynamoDB table. With a symbol it returns one row (404 when absent
or past `expires_at`); without one it scans a page of active rows and returns an
`X-Next-Cursor` header for pagination. This is what the UI heatmap reads.

### `api_ticker_suggest` — `GET /tickers/suggest?q=…`

Prefix autocomplete over the valid-ticker universe. Touches no DynamoDB: the universe
comes from SSM, an env var, or the JSON bundled in the image, cached in-process behind a
TTL because this endpoint is hit on every keystroke.

## Model promotion

### `model_promote`

Triggered by EventBridge when a model package in the FinSense package group moves to
`Approved` — the manual gate at the end of the training pipeline. It creates a SageMaker
model and endpoint config for that package version, then creates the endpoint on first
promotion or updates it in place afterwards, and prunes all but the last `KEEP_VERSIONS`
generations.

This Lambda, not Terraform, owns the model / endpoint config / endpoint. Both creates are
no-ops on redelivery, since EventBridge delivers at least once. Direct invoke with a
`model_package_arn` is the rollback path; an empty payload promotes the most recently
approved version.

## `finsense_shared`

The layer every image copies in. Handlers stay thin because the contracts live here.

| Module | Contents |
|--------|----------|
| `pipeline.py` | The S3 key layout, one builder + one validator per SQS queue, and the DynamoDB cache item shape — the contracts the stages agree on |
| `sentiment.py` | Reading class probabilities: confidence gating (`is_low_confidence`) and per-symbol aggregation |
| `sources/` | Source adapters producing text to score — `news_rss` (Google News), `reddit`, and `collect_for_symbol` which merges and de-duplicates them |
| `tickers/` | Symbol syntax (`normalize_symbol`), the run universe and the accepted universe, company names, and the bundled US ticker JSON |
| `aws/` | Thin boto3 wrappers: `s3` (JSON Lines read/write), `sqs` (send, consume, partial-batch failures), `sagemaker` (batched `InvokeEndpoint`) |
| `http.py` | API Gateway v2 request parsing and response shaping |
| `llm_label.py` | The provider-agnostic pseudo-labeler, with keys resolved from Secrets Manager |

## Building

Each `Dockerfile` copies `finsense_shared/` and its own `handler.py` into the Lambda task
root, with `src/lambdas` as the build context — so a Dockerfile is only valid when built
from that directory. `scripts/lambda-build-push.sh` does this for all nine locally; CI
does the same in a matrix. Both derive the tag from `scripts/image-tag.sh`, so images
built either way from the same commit are interchangeable.

`scripts/check_lambda_sources.py` compiles every file first: Docker's `COPY` is
byte-for-byte, so a file Python cannot parse builds and pushes cleanly and then fails at
Lambda import time.
