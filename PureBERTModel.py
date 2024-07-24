# We can try training our BERT model purely on the Financial Phrasebank dataset.
# Due to a limited amount of data, it may not be the most effective approach.
# However, it is a good starting point to understand how the model performs on this dataset.

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load our Financial Phrasebank dataset
data = pd.read_csv('FinancialPhraseBank-v1.0/Sentences_75Agree.txt', sep='\@', header=None)
X = data['text'].tolist()
y = data['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).tolist()

# Split our data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenize our data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)

# Prepare datasets
class FinancialDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
train_dataset = FinancialDataset(train_encodings, y_train)
val_dataset = FinancialDataset(val_encodings, y_val)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader):
        inputs = {key: val.to('cuda') for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f'Epoch {epoch + 1}, Val Loss: {val_loss:.4f}')

