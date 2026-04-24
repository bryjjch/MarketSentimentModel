"""Build the FinSense SageMaker Pipeline programmatically.

The pipeline implements the following DAG:

    DataPrep (Processing)
        -> MLM Training
        -> Classifier Training
        -> Evaluation (Processing)
        -> ConditionStep (macro_f1 >= threshold)
            -> RegisterModel (on pass)

The returned ``Pipeline`` object can be serialized to JSON for Terraform deployment
or upserted directly via the SageMaker API.
"""

from __future__ import annotations

from pathlib import Path

import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PIPELINE_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _PIPELINE_DIR / "scripts"
_ENTRY_POINTS_DIR = _PIPELINE_DIR / "entry_points"
_TRAINING_PKG_DIR = _REPO_ROOT / "src" / "training"

PIPELINE_NAME = "FinSenseSentimentPipeline"


def _resolve(p: Path) -> str:
    return str(p.resolve())


def build_pipeline(
    role: str,
    *,
    pipeline_name: str = PIPELINE_NAME,
    default_bucket: str | None = None,
    sagemaker_session: sagemaker.Session | None = None,
) -> Pipeline:
    """Construct and return the full SageMaker Pipeline object.

    Parameters
    ----------
    role:
        IAM role ARN used by training / processing jobs and model registration.
    pipeline_name:
        Name for the pipeline resource.
    default_bucket:
        S3 bucket for intermediate artifacts. Falls back to the session default.
    sagemaker_session:
        Optional pre-configured session (useful for local / offline JSON generation).
    """
    session = sagemaker_session or sagemaker.Session()
    bucket = default_bucket or session.default_bucket()

    # ------------------------------------------------------------------
    # Pipeline parameters
    # ------------------------------------------------------------------
    param_data_bucket = ParameterString(
        name="DataBucket",
        default_value=bucket,
    )
    param_curated_prefix = ParameterString(
        name="CuratedS3Prefix",
        default_value="curated/",
    )
    param_base_model = ParameterString(
        name="BaseModel",
        default_value="bert-base-uncased",
    )
    param_train_image = ParameterString(
        name="TrainImageUri",
        default_value="763104351884.dkr.ecr.us-east-1.amazonaws.com/"
        "huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04",
    )
    param_inference_image = ParameterString(
        name="InferenceImageUri",
        default_value="763104351884.dkr.ecr.us-east-1.amazonaws.com/"
        "huggingface-pytorch-inference:2.6.0-transformers4.51.3-cpu-py312-ubuntu22.04-",
    )
    param_processing_instance = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.large",
    )
    param_training_instance = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.g4dn.xlarge",
    )
    param_mlm_epochs = ParameterInteger(name="MlmEpochs", default_value=3)
    param_clf_epochs = ParameterInteger(name="ClfEpochs", default_value=3)
    param_model_package_group = ParameterString(
        name="ModelPackageGroup",
        default_value="finsense-sentiment",
    )
    param_macro_f1_threshold = ParameterFloat(
        name="MacroF1Threshold",
        default_value=0.80,
    )
    param_test_ratio = ParameterFloat(name="TestRatio", default_value=0.10)
    param_seed = ParameterInteger(name="Seed", default_value=42)

    pipeline_s3_root = f"s3://{bucket}/{pipeline_name}"

    # ------------------------------------------------------------------
    # Step 1 – Data preparation (Processing)
    # ------------------------------------------------------------------
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=param_processing_instance,
        instance_count=1,
        sagemaker_session=session,
    )

    data_prep_args = sklearn_processor.run(
        code=_resolve(_SCRIPTS_DIR / "prepare_training_data.py"),
        inputs=[
            ProcessingInput(
                source=f"s3://{param_curated_prefix}",
                input_name="curated",
                destination="/opt/ml/processing/input/curated",
                s3_data_distribution_type="FullyReplicated",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="mlm_corpus", source="/opt/ml/processing/output/mlm_corpus",
                             destination=f"{pipeline_s3_root}/data_prep/mlm_corpus"),
            ProcessingOutput(output_name="phrasebank_train", source="/opt/ml/processing/output/phrasebank_train",
                             destination=f"{pipeline_s3_root}/data_prep/phrasebank_train"),
            ProcessingOutput(output_name="pseudo_data", source="/opt/ml/processing/output/pseudo_data",
                             destination=f"{pipeline_s3_root}/data_prep/pseudo_data"),
            ProcessingOutput(output_name="test_data", source="/opt/ml/processing/output/test_data",
                             destination=f"{pipeline_s3_root}/data_prep/test_data"),
        ],
        arguments=[
            "--test-ratio", param_test_ratio.to_string(),
            "--seed", param_seed.to_string(),
        ],
    )

    step_data_prep = ProcessingStep(name="DataPrep", step_args=data_prep_args)

    # ------------------------------------------------------------------
    # Step 2 – MLM continued pre-training
    # ------------------------------------------------------------------
    mlm_estimator = HuggingFace(
        entry_point="run_mlm.py",
        source_dir=_resolve(_ENTRY_POINTS_DIR),
        dependencies=[_resolve(_TRAINING_PKG_DIR)],
        image_uri=param_train_image,
        role=role,
        instance_count=1,
        instance_type=param_training_instance,
        sagemaker_session=session,
        hyperparameters={
            "num_train_epochs": param_mlm_epochs,
            "model_name": param_base_model,
            "fp16": "True",
        },
        output_path=f"{pipeline_s3_root}/mlm",
    )

    mlm_train_args = mlm_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_data_prep.properties.ProcessingOutputConfig.Outputs[
                    "mlm_corpus"
                ].S3Output.S3Uri,
                content_type="application/jsonlines",
            ),
        },
    )

    step_mlm_train = TrainingStep(name="MLMPreTraining", step_args=mlm_train_args)

    # ------------------------------------------------------------------
    # Step 3 – Classifier fine-tuning
    # ------------------------------------------------------------------
    clf_estimator = HuggingFace(
        entry_point="run_classifier.py",
        source_dir=_resolve(_ENTRY_POINTS_DIR),
        dependencies=[_resolve(_TRAINING_PKG_DIR)],
        image_uri=param_train_image,
        role=role,
        instance_count=1,
        instance_type=param_training_instance,
        sagemaker_session=session,
        hyperparameters={
            "base_model": param_base_model,
            "num_train_epochs": param_clf_epochs,
            "test_ratio": 0,
            "fp16": "True",
            "metric_for_best_model": "macro_f1",
            "phrasebank_txt": "/opt/ml/input/data/phrasebank/Sentences_75Agree.txt",
            "pseudo_data": "/opt/ml/input/data/pseudo/pseudo.jsonl",
            "mlm_checkpoint": "/opt/ml/input/data/mlm_checkpoint",
        },
        output_path=f"{pipeline_s3_root}/classifier",
    )

    clf_train_args = clf_estimator.fit(
        inputs={
            "phrasebank": TrainingInput(
                s3_data=step_data_prep.properties.ProcessingOutputConfig.Outputs[
                    "phrasebank_train"
                ].S3Output.S3Uri,
            ),
            "pseudo": TrainingInput(
                s3_data=step_data_prep.properties.ProcessingOutputConfig.Outputs[
                    "pseudo_data"
                ].S3Output.S3Uri,
            ),
            "mlm_checkpoint": TrainingInput(
                s3_data=step_mlm_train.properties.ModelArtifacts.S3ModelArtifacts,
            ),
        },
    )

    step_clf_train = TrainingStep(name="ClassifierTraining", step_args=clf_train_args)

    # ------------------------------------------------------------------
    # Step 4 – Evaluation (Processing)
    # ------------------------------------------------------------------
    eval_processor = ScriptProcessor(
        command=["python3"],
        image_uri=param_train_image,
        role=role,
        instance_count=1,
        instance_type=param_processing_instance,
        sagemaker_session=session,
    )

    eval_property_file = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    eval_args = eval_processor.run(
        code=_resolve(_SCRIPTS_DIR / "evaluate_classifier.py"),
        inputs=[
            ProcessingInput(
                source=step_clf_train.properties.ModelArtifacts.S3ModelArtifacts,
                input_name="model",
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                source=step_data_prep.properties.ProcessingOutputConfig.Outputs[
                    "test_data"
                ].S3Output.S3Uri,
                input_name="test",
                destination="/opt/ml/processing/input/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output/evaluation",
                destination=f"{pipeline_s3_root}/evaluation",
            ),
        ],
    )

    step_evaluation = ProcessingStep(
        name="EvaluateClassifier",
        step_args=eval_args,
        property_files=[eval_property_file],
    )

    # ------------------------------------------------------------------
    # Step 5 – Quality gate (Condition)
    # ------------------------------------------------------------------
    macro_f1_value = JsonGet(
        step_name=step_evaluation.name,
        property_file=eval_property_file,
        json_path="metrics.macro_f1.value",
    )

    condition_macro_f1 = ConditionGreaterThanOrEqualTo(
        left=macro_f1_value,
        right=param_macro_f1_threshold,
    )

    # ------------------------------------------------------------------
    # Step 6 – Register model (only on pass)
    # ------------------------------------------------------------------
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{pipeline_s3_root}/evaluation/evaluation.json",
            content_type="application/json",
        ),
    )

    step_register = RegisterModel(
        name="RegisterClassifier",
        estimator=clf_estimator,
        model_data=step_clf_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.g4dn.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=param_model_package_group,
        model_metrics=model_metrics,
        image_uri=param_inference_image,
    )

    step_condition = ConditionStep(
        name="CheckMacroF1",
        conditions=[condition_macro_f1],
        if_steps=[step_register],
        else_steps=[],
    )

    # ------------------------------------------------------------------
    # Assemble pipeline
    # ------------------------------------------------------------------
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            param_data_bucket,
            param_curated_prefix,
            param_base_model,
            param_train_image,
            param_inference_image,
            param_processing_instance,
            param_training_instance,
            param_mlm_epochs,
            param_clf_epochs,
            param_model_package_group,
            param_macro_f1_threshold,
            param_test_ratio,
            param_seed,
        ],
        steps=[step_data_prep, step_mlm_train, step_clf_train, step_evaluation, step_condition],
        sagemaker_session=session,
    )

    return pipeline
