import numpy as np
import joblib
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

import sys
sys.path.insert(0, "..")

from staging import resolve_data_path, construct_data_path

from staging.dl_eval.predict_dl_sentiment import get_lstm_sentiment_prediction
from staging.dl_eval.predict_dl_sentiment import get_multiplicative_lstm_sentiment_prediction
from staging.dl_eval.predict_dl_sentiment import get_malstm_fcn_sentiment_prediction
from staging.dl_eval.predict_dl_ratings import get_lstm_ratings_prediction
from staging.dl_eval.predict_dl_ratings import get_multiplicative_lstm_ratings_prediction
from staging.dl_eval.predict_dl_ratings import get_malstm_fcn_ratings_prediction

from staging.utils.sklearn_utils import compute_metrics
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES, RATINGS_CLASS_NAMES

from staging.utils.keras_utils import prepare_yelp_ratings_dataset_keras, prepare_yelp_reviews_dataset_keras
from staging.utils.keras_utils import MAX_SEQUENCE_LENGTH, MAX_NB_WORDS


if __name__ == '__main__':
    # choose which predictor you would like ; default is logistic regression

    # predictor = get_lstm_sentiment_prediction
    predictor = get_multiplicative_lstm_sentiment_prediction
    # predictor = get_malstm_fcn_sentiment_prediction

    # Change the path here if needed by looking at the "data" directory
    path = "datasets/yelp-reviews/reviews.csv"
    path = resolve_data_path(path)

    print("Loading reviews dataset...")
    (data, labels, word_index) = prepare_yelp_reviews_dataset_keras(path, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    labels = np.argmax(labels, axis=-1)

    print("Dataset prepared ! Calculating sentiment scores")
    predicted_reviews = predictor(data, False)[0]

    compute_metrics(labels, predicted_reviews, SENTIMENT_CLASS_NAMES)

    # predictor = get_lstm_ratings_prediction
    # predictor = get_multiplicative_lstm_ratings_prediction
    predictor = get_malstm_fcn_ratings_prediction

    print("Loading ratings dataset...")
    (data, labels, word_index) = prepare_yelp_ratings_dataset_keras(path, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    min_rating = np.min(labels)
    labels = np.argmax(labels, axis=-1) + min_rating

    print("Dataset prepared ! Calculating ratings scores")
    predicted_ratings = predictor(data, False)[0] + np.min(labels)

    df = pd.DataFrame(data={
        'Labels': labels,
        'Predictions': predicted_ratings
    })

    df_path = 'results/yelp/dl_ratings_predictions.csv'
    df_path = construct_data_path(df_path)

    df.to_csv(df_path, encoding='utf-8', index_label='id')

    compute_metrics(labels, predicted_ratings, RATINGS_CLASS_NAMES)




