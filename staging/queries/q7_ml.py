import numpy as np
import joblib
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)

import sys
sys.path.insert(0, "..")

from staging import resolve_data_path, construct_data_path
from staging.ml_eval.predict_ml_ratings import get_decision_tree_rating_prediction
from staging.ml_eval.predict_ml_ratings import get_random_forest_rating_prediction
from staging.ml_eval.predict_ml_ratings import get_logistic_regression_rating_prediction
from staging.utils.text_utils import prepare_yelp_reviews_dataset_sklearn, prepare_yelp_ratings_dataset_sklearn
from staging.utils.sklearn_utils import compute_metrics
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES, RATINGS_CLASS_NAMES


if __name__ == '__main__':
    # choose which predictor you would like ; default is logistic regression

    # Change the path here if needed by looking at the "data" directory
    path = "datasets/yelp-reviews/reviews.csv"
    path = resolve_data_path(path)

    #predictor = get_decision_tree_rating_prediction
    #predictor = get_random_forest_rating_prediction
    predictor = get_logistic_regression_rating_prediction

    print("Loading ratings dataset...")
    data, labels = prepare_yelp_ratings_dataset_sklearn(path)

    print("Dataset prepared ! Calculating ratings scores")
    predicted_ratings = predictor(data, False)[0] + np.min(labels)

    df = pd.DataFrame(data={
        'Labels': labels,
        'Predictions': predicted_ratings
    })

    df_path = 'results/yelp/ml_ratings_predictions.csv'
    df_path = construct_data_path(df_path)

    df.to_csv(df_path, encoding='utf-8', index_label='id')

    compute_metrics(labels, predicted_ratings, RATINGS_CLASS_NAMES)




