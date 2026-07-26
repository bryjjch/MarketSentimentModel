FROM public.ecr.aws/lambda/python:3.12

COPY shared/finsense_shared/ ${LAMBDA_TASK_ROOT}/finsense_shared/

COPY pipeline_collect/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
