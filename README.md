# FinSense

Training code for financial **three-class sentiment** (negative, neutral, positive) using Hugging Face **Transformers** and PyTorch. You can fine-tune **BERT**-family checkpoints (including **FinBERT**), optionally continue pre-training with **masked language modeling (MLM)** on your own text, and optionally mix in **LLM pseudo-labeled** data.

## Requirements

- **Python** 3.10 or newer  
- **GPU** recommended for classifier and MLM training (CPU is possible but slow)  
- **Transformers 5.x** expects a recent **PyTorch** (2.4+).
- Dependencies are declared in `pyproject.toml` (PyTorch, Transformers, Datasets, scikit-learn, pandas, etc.)

## Repository layout

| Path | Purpose |
|------|---------|
| `src/training/` | Training package: data helpers, pseudo-labeling, MLM, classifier, **inference**, metrics, run manifests |
| `notebooks/bert_text_classification.ipynb` | Exploratory / teaching notebook aligned with the pipeline |
| `data/` | Default download location for Financial PhraseBank (created on first use) |
| `tests/` | Pytest suite |
| `requirements-training.txt` | Editable install (`pip install -r requirements-training.txt`) |
| `requirements/pinned-train.txt` | Pinned versions for reproducible train + dev environments |
| `requirements/pinned-serve.txt` | Minimal pins for inference-only images |

## Installation

From the repository root:

```bash
pip install -r requirements-training.txt
```

Or equivalently:

```bash
pip install -e .
```

For optional dev dependencies (pytest):

```bash
pip install -e ".[dev]"
```

Install a CUDA-enabled **PyTorch** build from [pytorch.org](https://pytorch.org/get-started/locally/) first if you want GPU training.

### Reproducible / pinned environments

For Docker, SageMaker, or CI, install the exact library set from `requirements/pinned-train.txt` (after installing a matching **torch** wheel for your CUDA/CPU target). For inference-only containers, use `requirements/pinned-serve.txt`.

## Label convention

Training and saved models use integer labels aligned with Financial PhraseBank style:

| Label ID | Sentiment |
|----------|-----------|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

FinBERT checkpoints use a different default class order; the classifier script remaps weights so saved artifacts use the table above.

## Inference (train/serve alignment)

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

Training saves a standard **Transformers** sequence-classification directory under `output_dir`. That layout is suitable for bundling (e.g. `model.tar.gz`) and hosting on **Amazon SageMaker** or any service that loads the same stack, ideally using **`SentimentPredictor`** (or equivalent) so tokenization matches training.

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
