import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification

def model_fn(model_dir):
    # Load the tokenizer and model from the model directory
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer

def input_fn(request_body, request_content_type='application/json'):
    # Preprocess the input data
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        texts = data['texts']
        return texts
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    model, tokenizer = model
    inputs = tokenizer(input_data, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).tolist()
    return predictions

def output_fn(prediction, content_type='application/json'):
    return json.dumps(prediction)