# FinSense

FinSense is a financial sentiment analysis stack. This repository contains a Python **training** package (Hugging Face Transformers and PyTorch), an **AWS** deployment (SageMaker, API Gateway, Lambdas, and S3 buckets), and a small **React** UI for heatmaps showcasing sentiment across stocks.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/training/` | Training package: data helpers, pseudo-labeling, MLM, classifier, **inference**, metrics, run manifests |
| `terraform/` | AWS resources (model + data buckets, SageMaker model/endpoint, API routes, Lambdas, DynamoDB, EventBridge, pipeline definition) |
| `src/sagemaker/` | SageMaker **serving** handler and **training pipeline** (build script, processing scripts, training entry points) |
| `src/lambdas/` | Lambda handlers (api inference, sentiment by symbol API, cache read, ingestion, ingestion-prediction, pseudo-label) and shared `finsense_shared` code layer |
| `frontend/` | React + TypeScript + Vite UI for cache list / heatmap |
| `notebooks/` | Exploratory / demonstration notebooks |
| `data/` | Default download location for Financial PhraseBank (created on first use) |
| `tests/` | Pytest suite |
| `requirements` | Necessary packages for different environments |

## Cloud deployment and pipelines

Provisioning (S3, SageMaker endpoint, HTTP API, Lambdas, daily ingestion → ingestion-prediction → pseudo-label flow, DynamoDB cache, SageMaker training pipeline resource) is documented in **`terraform/README.md`**, including:

- How to obtain and point Terraform at `model.tar.gz` from `finsense-train-classifier`
- API routes (`POST /predict`, `POST /sentiment/by-symbol`, `GET /sentiment/cache`, `GET /sentiment/cache/{symbol}`)
- Data layout under the data bucket (`raw/`, `predictions/`, `pseudo/`, `curated/`)
- Building and starting the SageMaker training pipeline

Lambda dependency layer setup and `terraform apply` live there as well.

## Label convention

Training and saved models use integer labels aligned with Financial PhraseBank style:

| Label ID | Sentiment |
|----------|-----------|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

FinBERT checkpoints use a different default class order; the classifier script remaps weights so saved artifacts use the table above.
