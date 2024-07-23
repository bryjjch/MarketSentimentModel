# We can pre-train our model on the sentiment140 dataset and then fine-tune it on the Financial Phrasebank dataset
# This is a risky move as Twitter phrases and posts are much different from financial news
# However, it is still worth experimenting with the effects of multi-dataset training
# We will use the Hugging Face Transformers library to train our model

# This script is used to train the model on both datasets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the sentiment140 dataset
data = pd.read_csv('Sentiment140.csv', encoding='ISO-8859-1', header=None)
data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Map labels to integer values (encoding the labels)
# The sentiment140 dataset now has 3 classes: 0, 2, 4
# 0 corresponds to negative sentiment, 2 corresponds to neutral sentiment, and 4 corresponds to positive sentiment
data['target'] = data['target'].map({0: 0, 2: 1, 4: 2})

# Rename target column to sentiment
data = data.rename(columns={'target': 'sentiment'})

# Keep only the sentiment and text columns
data = data[['sentiment', 'text']]

# Split data
X_train, X_val, y_train, y_val = train_test_split(data['text'], 
                                                  data['sentiment'], 
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=data['sentment'])

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True)

# Prepare datasetes
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = SentimentDataset(train_encodings, y_train.tolist())
val_dataest = SentimentDataset(val_encodings, y_val.tolist())

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataest
)

# Train the model
trainer.train()

# Then, we can fine-tune on the Financial Phrasebank dataset
# Read the financial phrasebank dataset .txt file
financial_data = pd.read_csv('FinancialPhraseBank-v1.0/Sentences_75Agree.txt', sep='\@', header=None)

# Convert Sentiment from "negative", "neutral", "positive" to 0, 1, 2
financial_data[1] = financial_data[1].map({'negative': 0, 'neutral': 1, 'positive': 2})
X_fin = financial_data['News Headline']
y_fin = financial_data['Sentiment']

# Split data into training and test sets
X_train_fin, X_val_fin, y_train_fin, y_val_fin = train_test_split(X_fin, y_fin, test_size=0.2)

# Tokenize our financial data
fin_encodings = tokenizer(X_fin.tolist(), truncation=True, padding=True)
val_fin_encodings = tokenizer(X_val_fin.tolist(), truncation=True, padding=True)

# Prepare our financial dataset
fin_dataset = SentimentDataset(fin_encodings, y_fin.tolist())

# Fine-tune the model on financial data
trainer.train_dataset = fin_dataset
trainer.train()

# Test our model on the validation set
model.eval()
with torch.inference():
    val_encodings = tokenizer(X_val_fin.tolist(), truncation=True, padding=True)
    val_dataset = SentimentDataset(val_encodings, y_val_fin.tolist())
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
    total = 0
    correct = 0
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy: {correct / total}')
