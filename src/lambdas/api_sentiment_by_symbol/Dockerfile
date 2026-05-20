FROM public.ecr.aws/lambda/python:3.12

COPY shared/finsense_shared/ ${LAMBDA_TASK_ROOT}/finsense_shared/
RUN pip install --no-cache-dir google-genai==1.73.1 openai==1.59.6

COPY api_sentiment_by_symbol/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
