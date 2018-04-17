import numpy as np
import joblib
import logging
logging.basicConfig(level=logging.INFO)

import sys
sys.path.insert(0, "..")

from staging import resolve_data_path
from staging.ml_eval.predict_ml_sentiment import get_decision_tree_sentiment_prediction
from staging.ml_eval.predict_ml_sentiment import get_random_forest_sentiment_prediction
from staging.ml_eval.predict_ml_sentiment import get_logistic_regression_sentiment_prediction
from staging.ml_eval.predict_ml_ratings import get_decision_tree_rating_prediction
from staging.ml_eval.predict_ml_ratings import get_random_forest_rating_prediction
from staging.ml_eval.predict_ml_ratings import get_logistic_regression_rating_prediction
from staging.utils.text_utils import prepare_yelp_reviews_dataset_sklearn, prepare_yelp_ratings_dataset_sklearn
from staging.utils.sklearn_utils import compute_metrics
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES, RATINGS_CLASS_NAMES


if __name__ == '__main__':
    # choose which predictor you would like ; default is logistic regression

    # predictor = get_decision_tree_sentiment_prediction
    # predictor = get_random_forest_sentiment_prediction
    predictor = get_logistic_regression_sentiment_prediction

    # Change the path here if needed by looking at the "data" directory
    path = "datasets/yelp-reviews/reviews.csv"
    path = resolve_data_path(path)

    print("Loading reviews dataset...")
    data, labels = prepare_yelp_reviews_dataset_sklearn(path)

    print("Dataset prepared ! Calculating sentiment scores")
    predicted_reviews = predictor(data, False)[0]

    compute_metrics(labels, predicted_reviews, SENTIMENT_CLASS_NAMES)

    #predictor = get_decision_tree_rating_prediction
    #predictor = get_random_forest_rating_prediction
    predictor = get_logistic_regression_rating_prediction

    print("Loading ratings dataset...")
    data, labels = prepare_yelp_ratings_dataset_sklearn(path)

    print("Dataset prepared ! Calculating ratings scores")
    predicted_ratings = predictor(data, False)[0] + np.min(labels)

    compute_metrics(labels, predicted_ratings, RATINGS_CLASS_NAMES)




