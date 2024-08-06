import boto3
import sagemaker
import json
import logging
from botocore.exceptions import ClientError

# Upload the file
s3_client = boto3.client('s3')
with open('aws_deployment/model.tar.gz', 'rb') as f:
    s3_client.upload_fileobj(f, 'sagemaker-us-east-1-891377265162', 'model_artifacts')
print('Model uploaded to S3')