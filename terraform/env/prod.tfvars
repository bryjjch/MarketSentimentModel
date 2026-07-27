# Production variable values, applied by .github/workflows/deploy.yml.
#
# image_tag is not set here; CI passes it with -var from scripts/image-tag.sh.

aws_region   = "us-east-1"
project_name = "finsense"

bucket_name      = "finsense-models-bucket"
data_bucket_name = "finsense-data-bucket"
endpoint_name    = "finsense-endpoint"

# --- Inference ---------------------------------------------------------------
# Hugging Face PyTorch inference DLC. Reaches the endpoint via the model package
# the training pipeline registers, not via a Terraform-managed model resource.
sagemaker_image_uri                    = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.51.3-cpu-py312-ubuntu22.04"
sagemaker_serverless_memory_size_in_mb = 3072
sagemaker_serverless_max_concurrency   = 5

# --- API ---------------------------------------------------------------------
apigateway_throttle_rate_limit  = 10
apigateway_throttle_burst_limit = 5

# --- Daily ingestion pipeline ------------------------------------------------
data_retention_days    = 90
ingestion_schedule     = "cron(0 13 * * ? *)"
ingestion_max_articles = 15

sagemaker_batch_size = 32
low_conf_top_prob    = 0.65
low_conf_margin      = 0.0

llm_provider = "google"
llm_model    = "gemini-3.1-flash-lite"

# --- Retraining --------------------------------------------------------------
# Disabled until a manually started run has been watched end to end. Flip to true
# to let the schedule drive it; approvals still gate what reaches the endpoint.
retrain_schedule_enabled       = false
retrain_schedule               = "cron(0 6 1 * ? *)"
retrain_training_instance_type = "ml.g4dn.xlarge"
pipeline_macro_f1_threshold    = 0.80
model_versions_to_keep         = 3
