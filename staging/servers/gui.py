from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

API_URL_SENTIMENT = 'http://35.193.72.219:9000/sentiment/dl'
API_URL_RATINGS = 'http://35.193.72.219:9001/rating/dl'


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/sentiment-page', methods=['POST'])
def sentiment():
    print("Message", request.form['message'])
    post_fields = {'model_name': 'lstm',
                   'query': request.form['message']}

    sentiment = requests.post(API_URL_SENTIMENT, json=post_fields)
    # rating = requests.post(API_URL_RATINGS, json=post_fields)

    sentiment = json.loads(sentiment.text)
    # rating = json.loads(rating.text)

    sentiment_class, sentiment_confidence = sentiment['sentiment'], sentiment['confidence']
    # rating_class, rating_confidence = int(rating['rating']) + 1, rating['confidence']

    response = {'polarity': sentiment_confidence, 'subjectivity': sentiment_class}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='localhost', port=9000, debug=True)
