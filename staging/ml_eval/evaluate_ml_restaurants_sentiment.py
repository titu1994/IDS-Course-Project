import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import logging
logging.basicConfig(level=logging.INFO)

from staging import resolve_data_path, construct_data_path

from staging.ml_eval.predict_ml_sentiment import get_decision_tree_sentiment_prediction
from staging.ml_eval.predict_ml_sentiment import get_random_forest_sentiment_prediction
from staging.ml_eval.predict_ml_sentiment import get_logistic_regression_sentiment_prediction
from staging.ml_eval.predict_ml_ratings import get_decision_tree_rating_prediction
from staging.ml_eval.predict_ml_ratings import get_random_forest_rating_prediction
from staging.ml_eval.predict_ml_ratings import get_logistic_regression_rating_prediction

from staging.utils.text_utils import load_dataset, add_sentiment_classes, remove_stopwords_punctuation, tokenize, tfidf
from staging.utils.sklearn_utils import compute_metrics
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES, RATINGS_CLASS_NAMES


def prepare_yelp_dataset_sklearn(path : str, nb_sentiment_classes : int = 2) -> (np.array, np.array):
    '''
    Loads the yelp reviews dataset for Scikit-learn models,
    prepares them by adding the class label and cleaning the
    reviews, tokenize and normalize them.

    Args:
        path: resolved path to the dataset
        nb_sentiment_classes: number of sentiment classes.
            Can be 2 or 3 only.

    Returns:
        a tuple of (data, labels)
    '''
    df = load_dataset(path)

    # calculate the classes of the dataset
    df = add_sentiment_classes(df, key='rating', nb_classes=nb_sentiment_classes)

    # clean the dataset of stopwords and punctuations
    df = remove_stopwords_punctuation(df, key='review', return_sentence=True)

    # extract the cleaned reviews and the classes
    reviews = df['review']
    labels = df['class'].values
    ratings = df['rating'].values
    restaurant_ids = df['restaurantID']

    tokens = tokenize(reviews)
    data = tfidf(tokens)

    return data, labels, ratings, restaurant_ids


if __name__ == '__main__':
    # choose which predictor you would like ; default is logistic regression

    # predictor = get_decision_tree_sentiment_prediction
    # predictor = get_random_forest_sentiment_prediction
    predictor = get_logistic_regression_sentiment_prediction

    # Change the path here if needed by looking at the "data" directory
    path = "datasets/yelp-reviews/reviews.csv"
    path = resolve_data_path(path)

    # this dataset must match the contain two cols - "restaurantID" and "name"
    restaurant_data_path = "datasets/yelp-reviews/restaurants.csv"
    restaurant_data_path = resolve_data_path(restaurant_data_path)

    df = pd.read_csv(restaurant_data_path, header=0)
    df = df[['restaurantID', 'name']]

    print("Loading reviews dataset...")
    data, labels, ratings, ids = prepare_yelp_dataset_sklearn(path)

    print("Dataset prepared ! Calculating sentiment scores")
    predicted_reviews = predictor(data, False)[0]

    # predictor = get_decision_tree_rating_prediction
    # predictor = get_random_forest_rating_prediction
    predictor = get_logistic_regression_rating_prediction

    print("Getting ratings scores")
    predicted_ratings = predictor(data, False)[0] + np.min(ratings)

    restaurant_names = []
    sentiment_prediction = []
    review_prediction = []

    for index in range(len(labels)):
        id = ids.iloc[index]  # get the restaurant id
        restaurant_name = df.loc[df['restaurantID'] == id, 'name'].values[0]  # get the restaurant name from the id

        restaurant_names.append(restaurant_name)
        sentiment_prediction.append(predicted_reviews[index])
        review_prediction.append(predicted_ratings[index])

    # build a dataframe and save it in a path
    df_path = "results/yelp/ml_sentiment_rating_query_result.csv"
    df_path = construct_data_path(df_path)

    dataframe = pd.DataFrame(data={
        'RestaurantName': restaurant_names,
        'SentimentLabels': sentiment_prediction,
        'ReviewRating': review_prediction
    }, columns=['RestaurantName', 'SentimentLabels', 'ReviewRating'])

    dataframe.to_csv(df_path, index=None)

    # compute the f1 scores of ratings here itself
    compute_metrics(labels, predicted_reviews, SENTIMENT_CLASS_NAMES)
    compute_metrics(ratings, predicted_ratings, RATINGS_CLASS_NAMES)

    """
    Query :
    
    Identify how the sentiments relate to the review rating by plotting average ratings against most frequent overall sentiments.
    Sentiment labels, Average Review Rating
    """
    sns.countplot(x='ReviewRating', hue='SentimentLabels', data=dataframe,
                  palette=sns.color_palette('RdBu', n_colors=2), saturation=1.0)
    plt.xlabel('Review Ratings')
    plt.ylabel('Instance Counts')
    plt.show()


