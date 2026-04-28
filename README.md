# FinSense

FinSense is a financial sentiment analysis stack. This repository contains a Python **training** package (Hugging Face Transformers and PyTorch), an optional **AWS** deployment (SageMaker, API Gateway, Lambdas, and S3 buckets), a **SageMaker Pipeline** definition for end-to-end retraining, and a small **React** UI for heatmaps showcasing sentiment across stocks.

## Requirements

- **Python** 3.10 or newer  
- **GPU** recommended for classifier and MLM training (CPU is possible but slow)  
- **Transformers 5.x** expects a recent **PyTorch** (2.4+).
- Dependencies are declared in `pyproject.toml` (PyTorch, Transformers, Datasets, scikit-learn, pandas, OpenAI / Google GenAI clients, etc.)

## Repository layout

| Path | Purpose |
|------|---------|
| `src/training/` | Training package: data helpers, pseudo-labeling, MLM, classifier, **inference**, metrics, run manifests |
| `infra/terraform/` | AWS resources (model + data buckets, SageMaker model/endpoint, API routes, Lambdas, DynamoDB, EventBridge, pipeline definition) |
| `infra/sagemaker/` | SageMaker **serving** handler and **training pipeline** (build script, processing scripts, training entry points) |
| `infra/lambda/` | Lambda handlers (predict, sentiment by symbol, cache read, ingestion, prediction, pseudo-label) and shared `finsense_shared` code layer |
| `web/` | React + TypeScript + Vite UI for cache list / heatmap |
| `notebooks/` | Exploratory / demonstration notebooks |
| `data/` | Default download location for Financial PhraseBank (created on first use) |
| `tests/` | Pytest suite |
| `requirements` | Necessary packages for different environments |

## Installation (training package)

From the repository root:

```bash
pip install -r requirements-training.txt
```

Or equivalently:

```bash
pip install -e .
```

For optional dev dependencies (pytest; **boto3** is included for tests that exercise AWS-related helpers):

```bash
pip install -e ".[dev]"
```

## Cloud deployment and pipelines

Provisioning (S3, SageMaker endpoint, HTTP API, Lambdas, daily ingestion → prediction → pseudo-label flow, DynamoDB cache, SageMaker training pipeline resource) is documented in **`infra/README.md`**, including:

- How to obtain and point Terraform at `model.tar.gz` from `finsense-train-classifier`
- API routes (`POST /predict`, `POST /sentiment/by-symbol`, `GET /sentiment/cache`, `GET /sentiment/cache/{symbol}`)
- Data layout under the data bucket (`raw/`, `predictions/`, `pseudo/`, `curated/`)
- Building and starting the SageMaker training pipeline

Lambda dependency layer setup and `terraform apply` live there as well.

## Web dashboard

From `web/`:

```bash
npm install
cp .env.example .env.local
# Set VITE_API_BASE_URL to your HTTP API invoke URL (see infra README / terraform outputs)
npm run dev
```

The app calls the deployed API (e.g. cached sentiment list and per-symbol cache). It does not run the training stack locally.

## Label convention

Training and saved models use integer labels aligned with Financial PhraseBank style:

| Label ID | Sentiment |
|----------|-----------|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

FinBERT checkpoints use a different default class order; the classifier script remaps weights so saved artifacts use the table above.

## Inference

```python
from training.inference import SentimentPredictor

p = SentimentPredictor("outputs/clf_finbert", device="cpu")
rows = p.predict(["EPS beat", "   ", "guidance cut"])
```

## Command-line tools

After installation, these entry points are available (see `--help` on each):

### 1. Fine-tune the classifier (`finsense-train-classifier`)

Uses **Financial PhraseBank** by default (downloaded into `data/` if missing). Writes a **Hugging Face model folder** plus **`training_manifest.json`** under `--output_dir`.

**Evaluation**: each validation epoch logs **accuracy**, **macro / weighted F1**, **macro precision/recall**, **per-class precision/recall/F1**, and **confusion matrix cells** (`cm_i_j`).

**Splits**: by default **10%** of rows are held out as a **test / production-eval** set (`--test_ratio 0.1`). The remaining data are split into train/validation using `--val_ratio` (default **0.2** of that remainder). Pass `--test_ratio 0` for the legacy behavior (train/val only on the full table).

**Model selection**: `--metric_for_best_model` (default `macro_f1`) controls `load_best_model_at_end`.

```bash
finsense-train-classifier --base_model ProsusAI/finbert --output_dir outputs/clf_finbert
finsense-train-classifier --base_model bert-base-uncased --mlm_checkpoint outputs/mlm_bert --pseudo_data data/pseudo.jsonl --output_dir outputs/clf_mlm
```

Useful flags include `--phrasebank_txt`, `--pseudo_data`, `--pseudo_weight`, `--num_train_epochs`, `--fp16` (CUDA), `--max_length`, `--test_ratio`, and `--metric_for_best_model`.

### 2. Continued pre-training with MLM (`finsense-train-mlm`)

Unlabeled **JSONL** (field `text` by default) and/or **`.txt`** (one document per line):

```bash
finsense-train-mlm --train_files data/wsb.jsonl data/reuters_lines.txt --output_dir outputs/mlm_bert
```

Under SageMaker `--train_files` defaults to every `.jsonl`/`.txt` file in the `train` channel (`SM_CHANNEL_TRAIN`), `--output_dir` defaults to `SM_MODEL_DIR`, and HF Trainer checkpoints are kept under `SM_OUTPUT_DATA_DIR/checkpoints` so they are excluded from the packaged `model.tar.gz`.

### 3. Pseudo-labeling with an LLM (`finsense-pseudo-label`)

Reads JSONL with a text field, appends `label` / `label_name` per row. Requires API keys for cloud providers:

- **OpenAI**: set `OPENAI_API_KEY` (default provider; default model `gpt-4o-mini`).
- **Google AI Studio**: set `GOOGLE_API_KEY` or `GEMINI_API_KEY`, use `--provider google` (default model `gemini-2.0-flash`).
- **Offline stub**: `--provider echo` (random labels for plumbing tests only).

```bash
finsense-pseudo-label --input data/raw_news.jsonl --output data/pseudo.jsonl
finsense-pseudo-label --provider google --input data/raw_news.jsonl --output data/pseudo.jsonl --resume
```

## Run manifest (`training_manifest.json`)

After each classifier run, metadata is written next to the weights:

- Git revision (if available), library versions, **inference API version**
- Full CLI configuration, row counts (phrasebank / pseudo / train / val / test)
- Final validation metrics (including confusion matrix), optional **test** metrics, and best-checkpoint info

## Artifacts and deployment

Training writes a deploy-ready **Transformers** sequence-classification directory (weights, tokenizer, `training_manifest.json`, and `code/inference.py` for SageMaker hosting) under `--output_dir`. Under SageMaker the script defaults `--output_dir` to `SM_MODEL_DIR` and keeps HF Trainer checkpoints under `SM_OUTPUT_DATA_DIR/checkpoints`, so the job's automatic `model.tar.gz` is ready to serve without any post-training packaging step. Locally, point `--output_dir` somewhere convenient and the same layout is produced.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

Integration tests that load a tiny Hugging Face model are **skipped** when Transformers reports PyTorch is unavailable (for example an older torch than 2.4 with Transformers 5.x). Upgrade PyTorch to run them.

## Development note

Run modules from the repo root with `PYTHONPATH=src` if you use `python -m training.train_classifier` without an editable install.

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m training.train_classifier --base_model ProsusAI/finbert --output_dir outputs/clf_finbert
```

Command Prompt (`cmd.exe`): `set PYTHONPATH=src` before the same `python -m ...` line. Linux or macOS: `export PYTHONPATH=src`.
