# BERT-based Sentiment Analysis for Financial News

## Overview

This project involves building a sentiment analysis system for financial news articles using a BERT-based model. The model predicts the sentiment (negative, neutral, positive) associated with a particular stock based on news articles. The system includes a web scraper to collect relevant news articles, a machine learning model to predict sentiment, and a web interface to display the results.

## Components
1. BERT Model
2. Web Scraping with SerpAPI
3. Web Application Interface
4. Deployment on AWS Sagemaker

## 1. BERT Model

### Description
BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art language representation model. We fine-tuned a pre-trained BERT model on the Financial PhraseBank dataset, which contains financial phrases labeled with sentiment categories (negative, neutral, positive).

### Training Details
- Dataset: Financial PhraseBank
- Pre-trained Model: `bert-base-uncased` from Hugging Face Transformers
- Fine-tuning: Adjusted for three sentiment classes
- Label Mapping:
    - Negative: 0
    - Neutral: 1
    - Positive: 2

## 2. Web Scraping with SerpAPI

### Description
The system uses SerpAPI to fetch news articles related to a specific stock symbol. SerpAPI provides a Google News search interface, enabling us to gather recent articles efficiently.

### Functionality
- Search Query: Takes a stock symbol and retrieves news articles related to that symbol.
- Timeout Handling: If an article isn't scraped within 20 seconds, the request is stopped to avoid hanging.

## 3. Web Application Interface

### Description

The web application allows users to enter a stock symbol and receive a sentiment score based on recent news articles. The sentiment scores are averaged from multiple articles to provide a comprehensive view.

### Features
- Input Field: For stock symbol
- Output: Sentiment score visualization

## 4. Deployment on AWS SageMaker

### Description

The model is deployed on AWS SageMaker, allowing scalable and secure hosting of the sentiment analysis service.

### Steps
1. Model Packaging: Bundle model artifacts and inference script into `model.tar.gz`.
2. Upload to S3: Store the package in an S3 bucket.
3. Deploy on SageMaker: Use the SageMaker SDK to create an endpoint for the model.

## Setup and Installation

### Prerequisites
- Python 3.6+
- PyTorch
- Transformers library from Hugging Face
- Flask
- AWS CLI and Boto3
- SerpAPI account and API key

### Installation
1. Clone the Repository:

```
git clone https://github.com/bryjjch/MarketSentimentModel.git
cd financial-sentiment-analysis
```

2. Install Dependencies:

```
pip install -r requirements.txt
```

3. Set Up Environment Variables:
- Store your SerpAPI key in `serp_api_key.py`.