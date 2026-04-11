# Market Sentiment Model

Training code for financial **three-class sentiment** (negative, neutral, positive) using Hugging Face **Transformers** and PyTorch. You can fine-tune **BERT**-family checkpoints (including **FinBERT**), optionally continue pre-training with **masked language modeling (MLM)** on your own text, and optionally mix in **LLM pseudo-labeled** data.


## Requirements

- **Python** 3.10 or newer  
- **GPU** recommended for classifier and MLM training (CPU is possible but slow)  
- Dependencies are declared in `pyproject.toml` (PyTorch, Transformers, Datasets, scikit-learn, pandas, etc.)

## Repository layout

| Path | Purpose |
|------|---------|
| `src/training/` | Training package: data helpers, pseudo-labeling, MLM, classifier fine-tuning |
| `notebooks/bert_text_classification.ipynb` | Exploratory / teaching notebook aligned with the pipeline |
| `data/` | Default download location for Financial PhraseBank (created on first use) |
| `requirements-training.txt` | Editable install (`pip install -r requirements-training.txt`) |

## Installation

From the repository root:

```bash
pip install -r requirements-training.txt
```

Or equivalently:

```bash
pip install -e .
```

Install a CUDA-enabled **PyTorch** build from [pytorch.org](https://pytorch.org/get-started/locally/) first if you want GPU training.

## Label convention

Training and saved models use integer labels aligned with Financial PhraseBank style:

| Label ID | Sentiment |
|----------|-----------|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

FinBERT checkpoints use a different default class order; the classifier script remaps weights so saved artifacts use the table above.

## Command-line tools

After installation, these entry points are available (see `--help` on each):

### 1. Fine-tune the classifier (`ms-train-classifier`)

Uses **Financial PhraseBank** by default (downloaded into `data/` if missing). Writes a **Hugging Face model folder** (config, weights, tokenizer) to `--output_dir`.

```bash
ms-train-classifier --base_model ProsusAI/finbert --output_dir outputs/clf_finbert
ms-train-classifier --base_model bert-base-uncased --mlm_checkpoint outputs/mlm_bert --pseudo_data data/pseudo.jsonl --output_dir outputs/clf_mlm
```

Useful flags include `--phrasebank_txt`, `--pseudo_data`, `--pseudo_weight`, `--num_train_epochs`, `--fp16` (CUDA), and `--max_length` (default 175).

### 2. Continued pre-training with MLM (`ms-train-mlm`)

Unlabeled **JSONL** (field `text` by default) and/or **`.txt`** (one document per line):

```bash
ms-train-mlm --train_files data/wsb.jsonl data/reuters_lines.txt --output_dir outputs/mlm_bert
```

### 3. Pseudo-labeling with an LLM (`ms-pseudo-label`)

Reads JSONL with a text field, appends `label` / `label_name` per row. Requires API keys for cloud providers:

- **OpenAI**: set `OPENAI_API_KEY` (default provider; default model `gpt-4o-mini`).
- **Google AI Studio**: set `GOOGLE_API_KEY` or `GEMINI_API_KEY`, use `--provider google` (default model `gemini-2.0-flash`).
- **Offline stub**: `--provider echo` (random labels for plumbing tests only).

```bash
ms-pseudo-label --input data/raw_news.jsonl --output data/pseudo.jsonl
ms-pseudo-label --provider google --input data/raw_news.jsonl --output data/pseudo.jsonl --resume
```

## Artifacts and deployment

Training saves a standard **Transformers** sequence-classification directory under `output_dir`. That layout is suitable for bundling (e.g. `model.tar.gz`) and hosting on **Amazon SageMaker** or any service that loads `AutoModelForSequenceClassification` + `AutoTokenizer`, provided your inference environment matches the libraries you used when saving.

## Development note

Run modules from the repo root with `PYTHONPATH=src` if you use `python -m training.train_classifier` without an editable install.

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m training.train_classifier --base_model ProsusAI/finbert --output_dir outputs/clf_finbert
```

Command Prompt (`cmd.exe`): `set PYTHONPATH=src` before the same `python -m ...` line. Linux or macOS: `export PYTHONPATH=src`.
