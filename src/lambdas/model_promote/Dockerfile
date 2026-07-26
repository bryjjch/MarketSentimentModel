FROM public.ecr.aws/lambda/python:3.12

# No finsense_shared copy: this function talks only to the SageMaker control plane
# and shares nothing with the ingestion pipeline.
COPY model_promote/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
