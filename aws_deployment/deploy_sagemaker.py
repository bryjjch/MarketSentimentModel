import sagemaker
model = sagemaker.model.Model(
    model_data='s3://sagemaker-us-east-1-891377265162/model_artifacts',
    image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.6.0-cpu-py36-r3.6',
    role='arn:aws:iam::891377265162:role/service-role/AmazonSageMaker-ExecutionRole-20240806T013326'
)

model.create(
    ExecutionRoleArn = 'arn:aws:iam::891377265162:role/service-role/AmazonSageMaker-ExecutionRole-20240806T013326',
    ModelName = 'model',
    PrimaryContainer = {
        'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.6.0-cpu-py36-r3.6',
        'ModelDataUrl': 's3://sagemaker-us-east-1-891377265162/model_artifacts'
    }
)

predictor = model.deploy(
    InitialInstanceCount = 1,
    InstanceType = 'ml.m4.large'
)