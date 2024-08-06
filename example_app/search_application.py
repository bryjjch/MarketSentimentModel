from flask import Flask, request, render_template
import requests
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import serpapi
import serp_api_key

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')\

app = Flask(__name__)

# Function to search for Google news articles
def search_google_news(query, num_results=10):
    output = []
    params = {
        "engine": "google_news",
        "q": query,
        "api_key": serp_api_key.API_KEY_SERVICE
    }
    result = serpapi.search(params)
    for article in result["news_results"]:
      output.append(article["title"])
    return result

# Preprocessing and prediction functions
def preprocess_articles(articles):
    encodings = tokenizer(articles, truncation=True, padding=True, max_length=128, return_tensors='pt')
    return encodings

def predict_sentiment(encodings):
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions

def decode_sentiment(predictions):
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return [sentiment_map[pred.item()] for pred in predictions]

# Create a heatmap
def create_heatmap(sentiment_labels):
    sentiment_counts = {'negative': 0, 'neutral': 0, 'positive': 0}
    for label in sentiment_labels:
        if label == 0:
            sentiment_counts['negative'] += 1
        elif label == 1:
            sentiment_counts['neutral'] += 1
        else:
            sentiment_counts['positive'] += 1

    data = [sentiment_counts['negative'], sentiment_counts['neutral'], sentiment_counts['positive']]
    labels = ['negative', 'neutral', 'positive']
    sns.heatmap([data], annot=True, fmt='d', cmap='coolwarm', xticklabels=labels)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    return img_base64

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    stock_symbol = request.form['stock_symbol']
    articles = search_google_news(f'{stock_symbol} stock')

    news_encodings = preprocess_articles([article for article in articles if article])
    predictions = predict_sentiment(news_encodings)
    sentiment_values = decode_sentiment(predictions)
    heatmap_base64 = create_heatmap(sentiment_values)
    
    average_sentiment = sum(sentiment_values) / len(sentiment_values) if sentiment_values else 0
    print(f'Overall Sentiment Score: {average_sentiment}')
    
    return render_template('result.html', heatmap_base64=heatmap_base64, stock_symbol=stock_symbol)

if __name__ == '__main__':
    app.run(debug=True)
