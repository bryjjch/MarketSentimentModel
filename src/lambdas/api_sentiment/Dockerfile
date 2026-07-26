FROM public.ecr.aws/lambda/python:3.12

COPY finsense_shared/ ${LAMBDA_TASK_ROOT}/finsense_shared/

COPY api_sentiment/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
