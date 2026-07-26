FROM public.ecr.aws/lambda/python:3.12

COPY finsense_shared/ ${LAMBDA_TASK_ROOT}/finsense_shared/
RUN pip install --no-cache-dir google-genai==1.73.1

COPY pipeline_label/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
