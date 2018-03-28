import numpy as np
import joblib

from staging import resolve_data_path
from staging.utils.text_utils import clean_text, tokenize, tfidf
from staging.utils.sklearn_utils import _get_predictions

# cache them to access faster multiple times
_decision_tree = None
_logistic_regression = None
_random_forest = None



def _preprocess_text(text):
    text = clean_text(text)
    text = ' '.join(text)
    texts = [text]
    tokens = tokenize(texts)
    tokens = tfidf(tokens)
    return tokens


def get_decision_tree_sentiment_prediction(text: str):
    global _decision_tree

    if _decision_tree is None:
        path = 'models/sklearn/sentiment/decision_tree.pkl'
        path = resolve_data_path(path)
        _decision_tree = joblib.load(path)

    tokens = _preprocess_text(text)

    pred = _get_predictions(_decision_tree, tokens)
    confidence = np.max(pred, axis=-1)[0]
    classification = np.argmax(pred, axis=-1)[0]

    return classification, confidence


def get_random_forest_sentiment_prediction(text: str):
    global _random_forest

    if _random_forest is None:
        path = 'models/sklearn/sentiment/random_forest.pkl'
        path = resolve_data_path(path)
        _random_forest = joblib.load(path)

    tokens = _preprocess_text(text)

    pred = _get_predictions(_random_forest, tokens)
    confidence = np.max(pred, axis=-1)[0]
    classification = np.argmax(pred, axis=-1)[0]

    return classification, confidence


def get_logistic_regression_sentiment_prediction(text: str):
    global _logistic_regression

    if _logistic_regression is None:
        path = 'models/sklearn/sentiment/logistic_regression.pkl'
        path = resolve_data_path(path)
        _logistic_regression = joblib.load(path)

    tokens = _preprocess_text(text)

    pred = _get_predictions(_logistic_regression, tokens)
    confidence = np.max(pred, axis=-1)[0]
    classification = np.argmax(pred, axis=-1)[0]

    return classification, confidence

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    text = "sad bad disgusting horrible"
    label, confidence = get_logistic_regression_sentiment_prediction(text)

    print("Class = ", label, "Confidence:", confidence)