# Training

The modeling code. Three CLIs — MLM pre-training, classifier fine-tuning, and LLM
pseudo-labeling — plus the shared helpers they lean on. The same code runs locally and
inside SageMaker training jobs; the only difference is where paths come from.

| Module | Purpose |
|--------|---------|
| `train_mlm.py` | Stage 1: masked-language-model continued pre-training on unlabeled financial text |
| `train_classifier.py` | Stage 2: 3-way sentiment fine-tuning, and the deploy-ready artifact |
| `pseudo_label.py` | Label unlabeled text with an LLM (offline counterpart to the `pipeline_label` Lambda) |
| `common.py` | Label conventions, PhraseBank download/parsing, labeled-table loading, SageMaker path resolution |
| `evaluation.py` | Flat scalar metrics — accuracy, macro/weighted F1, per-class, confusion cells |
| `inference.py` | `SentimentPredictor` for batch inference on a saved checkpoint |
| `artifacts.py` | `training_manifest.json`: args, metrics, library versions, git revision |

## Two stages: MLM, then the classifier

The pipeline is deliberately two training jobs rather than one.

**Stage 1 — MLM pre-training (`train_mlm.py`).** A BERT encoder pre-trained on general
English has never seen most of the vocabulary and phrasing that carries sentiment in
financial text. So before any classification head exists, the encoder is trained further
on the domain corpus with a plain masked-language-model objective: mask 15% of tokens,
predict them, no labels involved. This is why the corpus can be the *unlabeled* raw text
the daily pipeline has been accumulating — there is far more of it than there are labels.

The output is an encoder whose representations already separate financial language, saved
as a `BertForMaskedLM` checkpoint.

**Stage 2 — classifier fine-tuning (`train_classifier.py`).** A 3-way sequence
classification head is attached to the base model, and `--mlm_checkpoint` copies the
adapted encoder weights into its body — so fine-tuning starts from a domain-adapted
encoder rather than a general one. Then the labeled data does its job with a much smaller
number of examples than training from the general checkpoint would need.

```
unlabeled financial text ──► train_mlm ──► adapted encoder
                                                  │ --mlm_checkpoint
PhraseBank + pseudo-labels ─────────────────► train_classifier ──► model directory
```

Running stage 2 alone works fine — omit `--mlm_checkpoint` and it fine-tunes the base
model directly, which is the right thing when `--base_model ProsusAI/finbert` already
carries financial pre-training. The SageMaker pipeline always runs both, in order.

## Data and splits

`train_classifier.py` draws from two sources:

- **Financial PhraseBank** — the ground-truth corpus. Downloaded next to the module on
  first use, or read from `--phrasebank_txt` (which is how the SageMaker pipeline passes
  the copy staged from S3).
- **Pseudo-labeled rows** (`--pseudo_data`) — a CSV or JSONL of `text` + `label`. In the
  deployed system this is the `curated/` output of the daily pipeline: high-confidence
  model labels and LLM labels for the uncertain rows. `--pseudo_weight` up- or
  down-samples them relative to PhraseBank.

**Train/val/test splits are drawn from PhraseBank only.** A test fraction
(`--test_ratio`, default 0.1) is held out first, then the remainder is split into train and
validation (`--val_ratio`, default 0.2), both stratified by label when every class has at
least two rows. Pseudo-labeled rows are appended to the **training pool only** — never to
validation or test. That is the point: the system's own labels can improve the model
without contaminating the measurement of whether they did.

The SageMaker pipeline passes `test_ratio=0` here, because its DataPrep step has already
carved out a frozen test split that a separate evaluation step scores.

## Label convention

Integer labels follow Financial PhraseBank, and so does everything the system saves:

| Label | Sentiment |
|-------|-----------|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

FinBERT checkpoints ship with a different class order (0=positive, 1=negative,
2=neutral). When `--base_model` is such a checkpoint, training detects it from the model
config, remaps the dataset labels into FinBERT's order so the pre-trained head stays
useful, and then permutes the head's rows back before saving. Saved artifacts always use
the table above, so serving code never needs to know which base model was used.

## What a run produces

`--output_dir` is a deploy-ready Hugging Face model directory:

- the weights and tokenizer;
- `code/inference.py`, copied from `src/sagemaker/serving/` so the inference DLC finds a
  handler inside `model.tar.gz`;
- `training_manifest.json` — the arguments, validation and test metrics, confusion
  matrices, library versions and git revision for the run.

`--max_length` defaults to 175 and must match the serving handler; the two are pinned to
the same constant in `inference.py` for exactly that reason.

Under SageMaker the defaults do the right thing without any extra flags: `--output_dir`
falls back to `SM_MODEL_DIR` (which SageMaker packs into `model.tar.gz`), Trainer
checkpoints and logs go to `SM_OUTPUT_DATA_DIR/checkpoints` so they stay out of the model
artifact, and `train_mlm.py` picks up every `.jsonl`/`.txt` staged in the `train` channel.

## Running locally

```bash
pip install -e ".[train,dev]"

# Stage 1
python -m training.train_mlm \
  --train_files data/raw_news.jsonl --output_dir outputs/mlm_bert

# Stage 2, warm-started from stage 1
python -m training.train_classifier \
  --base_model bert-base-uncased \
  --mlm_checkpoint outputs/mlm_bert \
  --pseudo_data data/pseudo.jsonl \
  --output_dir outputs/clf_mlm

# Or straight from FinBERT, no MLM stage
python -m training.train_classifier \
  --base_model ProsusAI/finbert --output_dir outputs/clf_finbert

# Pseudo-label unlabeled text offline
python -m training.pseudo_label \
  --provider google --model gemini-3.1-flash-lite \
  --input data/raw_news.jsonl --output data/pseudo.jsonl
```

All three are also installed as console scripts: `finsense-train-mlm`,
`finsense-train-classifier`, `finsense-pseudo-label`. `pseudo_label.py` needs
`GOOGLE_API_KEY`/`GEMINI_API_KEY` or `OPENAI_API_KEY`, unless you use `--provider echo`,
a deterministic offline stub for testing the plumbing.

Local runs are for iteration only. Nothing that reaches the endpoint is trained here — the
SageMaker pipeline is the path to a registered model.
