import numpy as np
import joblib
from typing import Union

from staging import resolve_data_path
from staging.utils.text_utils import clean_text, tokenize, tfidf
from staging.utils.sklearn_utils import _get_predictions

# cache them to access faster multiple times
_decision_tree_ratings = None
_logistic_regression_ratings = None
_random_forest_ratings = None


def _initialize():
    global _decision_tree_ratings, _logistic_regression_ratings, _random_forest_ratings

    initialization_text = "default"
    _ = _preprocess_text(initialization_text)  # will initialize the tokenizer

    if _decision_tree_ratings is None:
        path = 'models/sklearn/ratings/decision_tree.pkl'
        path = resolve_data_path(path)
        _decision_tree_ratings = joblib.load(path)

    if _random_forest_ratings is None:
        path = 'models/sklearn/ratings/random_forest.pkl'
        path = resolve_data_path(path)
        _random_forest_ratings = joblib.load(path)

    if _logistic_regression_ratings is None:
        path = 'models/sklearn/ratings/logistic_regression.pkl'
        path = resolve_data_path(path)
        _logistic_regression_ratings = joblib.load(path)

    print("Initialized machine learning models !")


def _preprocess_text(text: str):
    text = clean_text(text)
    text = ' '.join(text)
    texts = [text]
    tokens = tokenize(texts)
    tokens = tfidf(tokens)
    return tokens


def get_decision_tree_rating_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _decision_tree_ratings

    if _decision_tree_ratings is None:
        path = 'models/sklearn/ratings/decision_tree.pkl'
        path = resolve_data_path(path)
        _decision_tree_ratings = joblib.load(path)

    if preprocess:
        tokens = _preprocess_text(text)
    else:
        tokens = text

    pred = _get_predictions(_decision_tree_ratings, tokens)
    confidence = np.max(pred, axis=-1)
    classification = np.argmax(pred, axis=-1)

    return classification, confidence


def get_random_forest_rating_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _random_forest_ratings

    if _random_forest_ratings is None:
        path = 'models/sklearn/ratings/random_forest.pkl'
        path = resolve_data_path(path)
        _random_forest_ratings = joblib.load(path)

    if preprocess:
        tokens = _preprocess_text(text)
    else:
        tokens = text

    pred = _get_predictions(_random_forest_ratings, tokens)
    confidence = np.max(pred, axis=-1)
    classification = np.argmax(pred, axis=-1)

    return classification, confidence


def get_logistic_regression_rating_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _logistic_regression_ratings

    if _logistic_regression_ratings is None:
        path = 'models/sklearn/ratings/logistic_regression.pkl'
        path = resolve_data_path(path)
        _logistic_regression_ratings = joblib.load(path)

    if preprocess:
        tokens = _preprocess_text(text)
    else:
        tokens = text

    pred = _get_predictions(_logistic_regression_ratings, tokens)

    confidence = np.max(pred, axis=-1)
    classification = np.argmax(pred, axis=-1)

    return classification, confidence

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    text = "sad bad disgusting horrible"
    label, confidence = get_logistic_regression_rating_prediction(text)

    print("Class = ", label, "Confidence:", confidence)