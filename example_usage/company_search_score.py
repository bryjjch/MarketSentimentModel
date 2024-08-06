# Import statements
import serpapi
import requests
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Set device to whatever device is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

# Step 1: Search Google News
def search_google_news(query, num_results=10):
    print(f"Searching Google News for: {query}")
    output = []
    params = {
        "engine": "google_news",
        "q": query,
        "api_key": "7d2578c0d84947939e1b3aab051d37a70c90e91a9ae0f3c2f98001559aba08f5"
    }
    print("Calling SerpAPI")
    result = serpapi.search(params)
    print("Going through articles")
    for article in result["news_results"]:
      output.append(article["title"])
    return result

# Step 2: Preprocess and Predict Sentiment
def preprocess_articles(articles):
    print("Preprocessing articles...")
    encodings = tokenizer(articles, truncation=True, padding=True, max_length=128, return_tensors='pt')
    return encodings

def predict_sentiment(encodings):
    print("Predicting sentiment...")
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions

def decode_sentiment(predictions):
    print("Decoding sentiment predictions...")
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return [sentiment_map[pred.item()] for pred in predictions]

stock_symbol = 'NVDA'  # Example stock symbol
news_titles = search_google_news(f'{stock_symbol} stock')

print(news_titles)
# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
state_dict = torch.load('Finetuned_BERT.pth', map_location=torch.device(device))
model.load_state_dict(state_dict)
model.eval()

# Preprocess articles
news_encodings = preprocess_articles([article for article in news_titles if article])

# Predict sentiment
scores = []
for article in news_encodings:
    predict_sentiment(article)
    scores.append(predict_sentiment(article))

# Decode and calculate average sentiment score
average_sentiment = sum(scores) / len(scores) if scores else 0  # Handle case with no valid scores
print(f'Overall Sentiment Score: {average_sentiment}')