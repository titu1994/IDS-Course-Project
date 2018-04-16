import numpy as np
import joblib
from typing import Union

from staging import resolve_data_path
from staging.utils.text_utils import clean_text, tokenize, tfidf
from staging.utils.sklearn_utils import _get_predictions

# cache them to access faster multiple times
_decision_tree_sentiment = None
_logistic_regression_sentiment = None
_random_forest_sentiment = None


def _initialize():
    global _decision_tree_sentiment, _logistic_regression_sentiment, _random_forest_sentiment

    initialization_text = "default"
    initialization_text = _preprocess_text(initialization_text)  # will initialize the tokenizer

    if _decision_tree_sentiment is None:
        path = 'models/sklearn/sentiment/decision_tree.pkl'
        path = resolve_data_path(path)
        _decision_tree_sentiment = joblib.load(path)

    if _random_forest_sentiment is None:
        path = 'models/sklearn/sentiment/random_forest.pkl'
        path = resolve_data_path(path)
        _random_forest_sentiment = joblib.load(path)

    if _logistic_regression_sentiment is None:
        path = 'models/sklearn/sentiment/logistic_regression.pkl'
        path = resolve_data_path(path)
        _logistic_regression_sentiment = joblib.load(path)

    print("Initialized machine learning models !")


def _preprocess_text(text: str):
    text = clean_text(text)
    text = ' '.join(text)
    texts = [text]
    tokens = tokenize(texts)
    tokens = tfidf(tokens)
    return tokens


def get_decision_tree_sentiment_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _decision_tree_sentiment

    if _decision_tree_sentiment is None:
        path = 'models/sklearn/sentiment/decision_tree.pkl'
        path = resolve_data_path(path)
        _decision_tree_sentiment = joblib.load(path)

    if preprocess:
        tokens = _preprocess_text(text)
    else:
        tokens = text

    pred = _get_predictions(_decision_tree_sentiment, tokens)
    confidence = np.max(pred, axis=-1)
    classification = np.argmax(pred, axis=-1)

    return classification, confidence


def get_random_forest_sentiment_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _random_forest_sentiment

    if _random_forest_sentiment is None:
        path = 'models/sklearn/sentiment/random_forest.pkl'
        path = resolve_data_path(path)
        _random_forest_sentiment = joblib.load(path)

    if preprocess:
        tokens = _preprocess_text(text)
    else:
        tokens = text

    pred = _get_predictions(_random_forest_sentiment, tokens)
    confidence = np.max(pred, axis=-1)
    classification = np.argmax(pred, axis=-1)

    return classification, confidence


def get_logistic_regression_sentiment_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _logistic_regression_sentiment

    if _logistic_regression_sentiment is None:
        path = 'models/sklearn/sentiment/logistic_regression.pkl'
        path = resolve_data_path(path)
        _logistic_regression_sentiment = joblib.load(path)

    if preprocess:
        tokens = _preprocess_text(text)
    else:
        tokens = text

    pred = _get_predictions(_logistic_regression_sentiment, tokens)
    confidence = np.max(pred, axis=-1)
    classification = np.argmax(pred, axis=-1)

    return classification, confidence

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    text = "sad bad disgusting horrible"
    label, confidence = get_logistic_regression_sentiment_prediction(text)

    print("Class = ", label, "Confidence:", confidence)