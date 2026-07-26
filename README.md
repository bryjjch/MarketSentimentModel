# FinSense

FinSense is a financial sentiment analysis stack built around a fine-tuned FinBERT classifier. It covers model training, AWS deployment, and a React UI for per-symbol sentiment heatmaps.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/training/` | Training package: data helpers, pseudo-labeling, MLM, classifier, inference, metrics, run manifests |
| `terraform/` | AWS resources (buckets, SageMaker endpoint, API Gateway, Lambdas, DynamoDB, EventBridge, pipeline) |
| `src/sagemaker/` | SageMaker serving handler and training pipeline (build script, processing scripts, training entry points) |
| `src/lambdas/` | Lambda handlers and shared `finsense_shared` code layer |
| `frontend/` | React + TypeScript + Vite UI for cache list / heatmap |
| `notebooks/` | Exploratory / demonstration notebooks |
| `data/` | Default download location for Financial PhraseBank (created on first use) |
| `tests/` | Pytest suite |

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
