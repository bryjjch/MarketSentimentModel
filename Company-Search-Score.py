from googlesearch import search
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# Step 1: Search Google News
def search_google_news(query, num_results=10):
    print(f"Searching Google News for: {query}")
    search_query = f'{query} site:news.google.com'
    results = search(search_query, num_results=num_results)
    return results

# Step 2: Scrape News Articles
def scrape_article(url, timeout=10):
    print(f"Scraping article: {url}")
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # grab the title of the article
        title = soup.find('h1').get_text()
        # return the title
        article = title
        print(f"Scraped article: {article[:50]}...")
        return article
    except requests.exceptions.Timeout:
        print(f"Timeout while scraping {url}")
        return ""
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Step 3: Preprocess and Predict Sentiment
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



# Main script
if __name__ == "__main__":
    stock_symbol = 'AAPL'  # Example stock symbol
    news_urls = search_google_news(f'{stock_symbol} stock')

    # Scrape articles with timeout using concurrent.futures
    articles = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_article, url): url for url in news_urls}
        for future in as_completed(future_to_url, timeout=20):  # Set the overall timeout for all tasks
            url = future_to_url[future]
            try:
                article = future.result(timeout=20)  # Set timeout for each individual task
                articles.append(article)
            except TimeoutError:
                print(f"Timeout while scraping {url}")
            except Exception as e:
                print(f"Error scraping {url}: {e}")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    state_dict = torch.load('path_to_your_model_state_dict.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Preprocess articles
    news_encodings = preprocess_articles([article for article in articles if article])

    # Predict sentiment
    predictions = predict_sentiment(news_encodings)

    # Decode and calculate average sentiment score
    sentiment_labels = decode_sentiment(predictions)
    # Convert sentiment labels to scores
    sentiment_scores = {'negative': 0, 'neutral': 1, 'positive': 2}
    scores = [sentiment_scores[label] for label in sentiment_labels]
    average_sentiment = sum(scores) / len(scores) if scores else 0  # Handle case with no valid scores
    print(f'Overall Sentiment Score: {average_sentiment}')
