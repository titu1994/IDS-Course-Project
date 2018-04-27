import json
from requests import post

API_URL_SENTIMENT = 'http://35.193.72.219:9000/sentiment/dl'
API_URL_RATINGS = 'http://35.193.72.219:9001/rating/dl'

post_fields = {'model_name': 'lstm',
               'query': ''}

print("Input : ", end='')
query = input()

while query.lower() != 'quit' or query.lower() == 'exit':
    post_fields['query'] = query

    sentiment = post(API_URL_SENTIMENT, json=post_fields)
    rating = post(API_URL_RATINGS, json=post_fields)

    print("Sentiment : ", sentiment.text)
    print("Rating : ", rating.text)

    print("Input : ", end='')
    query = input()
