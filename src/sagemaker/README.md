# SageMaker

Two folders, for the two sides of the model's life in AWS: `pipeline/` produces a model,
`serving/` runs one.

| Folder | Purpose |
|--------|---------|
| `pipeline/` | The training pipeline: the DAG definition, the CLI that compiles and upserts it, the processing scripts each step runs, and the entry points that wrap the training package |
| `serving/` | The inference handler that ships inside `model.tar.gz` and runs on the endpoint |

> **Import note:** the installed SageMaker SDK owns the top-level `sagemaker` name, so this
> directory must be on `PYTHONPATH` and its package imported as plain `pipeline`, never
> `sagemaker.pipeline`. That is why every invocation sets `PYTHONPATH=src/sagemaker`.

## `pipeline/`

| File | Role |
|------|------|
| `pipeline_definition.py` | `build_pipeline()` — constructs the whole DAG, its parameters and its quality gate |
| `build_pipeline.py` | CLI wrapper: compile the definition to JSON, or `--upsert` it into SageMaker |
| `scripts/prepare_training_data.py` | Processing step: assembles training channels from `curated/` and PhraseBank |
| `scripts/evaluate_classifier.py` | Processing step: scores the trained model on the held-out test split |
| `entry_points/run_mlm.py` | Training step entry point → `training.train_mlm` |
| `entry_points/run_classifier.py` | Training step entry point → `training.train_classifier`, after unpacking the MLM channel |

### How the pipeline operates

```
DataPrep (Processing)
   ├── mlm_corpus ──────────► MLMPreTraining (Training)
   │                                 │ model.tar.gz
   ├── phrasebank_train ───┐         ▼
   ├── pseudo_data ────────┴──► ClassifierTraining (Training)
   │                                 │ model.tar.gz
   └── test_data ──────────────► EvaluateClassifier (Processing)
                                     │ evaluation.json
                                     ▼
                              CheckMacroF1 (Condition)
                                     │ macro_f1 >= MacroF1Threshold
                                     ▼
                              RegisterClassifier → model package, PendingManualApproval
```

**DataPrep** reads two inputs: the `curated/` prefix of the data bucket — everything the
daily ingestion pipeline has labeled, both high-confidence model labels and LLM
pseudo-labels — and the seeded Financial PhraseBank corpus. It writes four channels: an
unlabeled text corpus for MLM, the PhraseBank train split in its original format, the
curated rows as labeled JSONL for the classifier, and a held-out PhraseBank test split
that no training step ever sees.

**MLMPreTraining** adapts the encoder to financial language before any classification head
exists (see `src/training/README.md` for why this ordering matters). Its output artifact is
fed to the next step as a training channel.

**ClassifierTraining** fine-tunes on PhraseBank plus the curated rows, starting from the
MLM encoder. Because SageMaker delivers a channel'd model artifact as a `model.tar.gz`,
`entry_points/run_classifier.py` extracts it — rejecting any member path that escapes the
destination — and rewrites `--mlm_checkpoint` to point at the extracted directory.

**EvaluateClassifier** loads the artifact, runs the held-out test split through it, and
writes `evaluation.json`. That file is declared as a `PropertyFile`, which is what makes
its metrics readable by the next step.

**CheckMacroF1** is a `ConditionStep`. It pulls `metrics.macro_f1.value` out of the
property file with `JsonGet` and compares it to the `MacroF1Threshold` parameter (0.80).
If it passes, **RegisterClassifier** registers a model package — carrying the inference
image URI and the evaluation metrics — in the model package group. If it fails, the branch
is empty: nothing is registered, and the endpoint is untouched.

Registration is not deployment. Every package lands as `PendingManualApproval`; approving
it is what triggers the `model_promote` Lambda to roll the endpoint forward.

### Parameters

Everything a run might want to vary is a pipeline parameter, so it can be overridden per
execution without recompiling: `DataBucket`, `CuratedS3Prefix`, `PhraseBankS3Prefix`,
`BaseModel`, `TrainImageUri`, `InferenceImageUri`, the two instance types, `MlmEpochs`,
`ClfEpochs`, `ModelPackageGroup`, `MacroF1Threshold`, `TestRatio` and `Seed`.

`TrainImageUri` and `InferenceImageUri` default to AWS Deep Learning Containers. Nothing
from `pyproject.toml` is installed in the cloud — the DLCs supply torch and transformers,
which is why those two URIs are the real dependency pins for training and serving.

### Building and upserting

The definition is not a Terraform resource. Compiling it uploads `sourcedir.tar.gz` to S3
and embeds the pipeline role ARN Terraform creates, so it can only be built *after* an
apply. CI does this as a post-apply job on every push to `main`, and `upsert` is
idempotent:

```bash
PYTHONPATH=src/sagemaker python -m pipeline.build_pipeline \
  --role "$(terraform -chdir=terraform output -raw pipeline_role_arn)" \
  --pipeline-name "$(terraform -chdir=terraform output -raw pipeline_name)" \
  --bucket "$(terraform -chdir=terraform output -raw data_bucket_name)" \
  --upsert
```

`--output <file>` writes the compiled JSON instead of upserting, for inspecting or diffing
a definition before it goes live. It still needs AWS credentials, because the SDK uploads
code artifacts while compiling.

Executions are started from the SageMaker console or by the EventBridge retrain schedule —
never by CI, and never locally.

## `serving/`

`serving/code/inference.py` is the endpoint's handler. It is copied into the model
directory during classifier training, so it travels inside `model.tar.gz` and the
inference DLC picks it up automatically.

It mirrors `training.inference.SentimentPredictor` — same `max_length` of 175, same
empty-text handling, same label IDs, same record shape — without importing the training
package, because nothing but the model artifact is present at serving time. The two must
be kept in step: `src/lambdas/finsense_shared/sentiment.py` reads the records this handler
returns.
