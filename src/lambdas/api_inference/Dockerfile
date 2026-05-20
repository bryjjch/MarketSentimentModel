FROM public.ecr.aws/lambda/python:3.12

COPY api_inference/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
