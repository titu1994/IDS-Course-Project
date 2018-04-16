import numpy as np
import pandas as pd
import os
import logging
import textblob
from typing import Union

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import load_dataset, add_sentiment_classes


def prepare_optim_dataframe(path: str, nb_sentiment_classes : int = 2):
    '''
    Loads the yelp reviews dataset, prepares them by
    adding the class label and then extracting the required items
    and storing them into a dataframe for faster loading next time.

    Args:
        path: resolved path to the dataset
        nb_sentiment_classes: number of sentiment classes.
            Can be 2 or 3 only.
    '''

    df = load_dataset(path)

    # calculate the classes of the dataset
    df = add_sentiment_classes(df, key='rating', nb_classes=nb_sentiment_classes)

    # extract the cleaned reviews and the classes
    reviews = df['review'].values
    labels = df['class'].values
    ratings = df['rating'].values
    restaurant_ids = df['restaurantID'].values

    percentage = len(reviews) // 100
    polarity_scores = []
    for i, review in enumerate(reviews):
        blob = textblob.TextBlob(review)
        polarity = blob.sentiment.polarity

        polarity_scores.append(polarity)

        if i % percentage == 0:
            logging.info("%d / 100%% sentiment text analyzed" % (int(i * 100 / len(reviews))))


    df = pd.DataFrame(data={
        'review': reviews,
        'class': labels,
        'rating': ratings,
        'restaurantID': restaurant_ids,
        'polarity': polarity_scores
    }, columns=['restaurantID','review', 'class', 'rating', 'polarity'])

    path = "datasets/optimization/yelp_data.csv"
    path = construct_data_path(path)

    df.to_csv(path, encoding='utf-8')
    print("Dataframe constructed at %s" % (path))


def load_optimization_data(path: str = "datasets/optimization/yelp_data.csv"):
    '''
    Loads the dataframe which was generated with the sentiment polarities for
    direct usage.

    Returns:
        a tuple of (text, class, rating, restaurant_id, polarity)
    '''
    path = resolve_data_path(path)

    df = pd.read_csv(path, header=0, encoding='utf-8')

    reviews = df['review'].values
    labels = df['class'].values
    ratings = df['rating'].values - 1
    restaurant_ids = df['restaurantID'].values
    polarity = df['polarity'].values

    return (reviews, labels, ratings, restaurant_ids, polarity)


def predict_sentiment(texts: Union[str, np.ndarray]) -> (np.ndarray):
    '''
    Returns the sentiment analysis of positive or negative rating

    Args:
        texts: can be a single string or a list of strings

    Returns:
        a list of sentiment ratings
    '''
    path = "models/optimization/sentiment_threshold.txt"
    path = resolve_data_path(path)

    with open(path, 'r') as f:
        threshold = float(f.read())

    if type(texts) == str:
        texts = [texts]

    labels = []
    for text in texts:
        blob = textblob.TextBlob(text)
        polarity = blob.sentiment.polarity

        label = polarity > threshold
        labels.append(label)

    labels = np.array(labels)
    return labels


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # change path if necessary inside the data folder
    path = "datasets/yelp-reviews/reviews.csv"
    path = resolve_data_path(path)

    prepare_optim_dataframe(path)

    reviews, labels, ratings, restaurant_ids, polarity = load_optimization_data()


