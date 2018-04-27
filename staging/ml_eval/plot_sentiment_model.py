import numpy as np
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')

from sklearn.metrics import confusion_matrix, f1_score

import sys
sys.path.insert(0, "..")

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import prepare_yelp_reviews_dataset_sklearn
from staging.utils.sklearn_utils import create_train_test_set, compute_metrics, make_f1_scorer
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES

from staging.ml_eval.predict_ml_sentiment import get_logistic_regression_sentiment_prediction
from staging.ml_eval.predict_ml_sentiment import get_random_forest_sentiment_prediction
from staging.ml_eval.predict_ml_sentiment import get_decision_tree_sentiment_prediction


def plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    f1 = f1_score(y_true, y_pred, [0, 1], average='micro')

    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 6))

    print(conf_matrix)
    df = pd.DataFrame(conf_matrix, index=SENTIMENT_CLASS_NAMES, columns=SENTIMENT_CLASS_NAMES)

    g = sns.heatmap(df, annot=True, annot_kws={"size": 16}, fmt='g', square=True)
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=12)
    g.set_xticklabels(g.get_xticklabels(), rotation=0, fontsize=12)

    title = model_name + " (F1 : %1.5f)" % f1
    plt.title(title, fontsize=16)
    plt.show()


if __name__ == '__main__':
    reviews_path = resolve_data_path('datasets/yelp-reviews/reviews.csv')
    data, labels = prepare_yelp_reviews_dataset_sklearn(reviews_path, nb_sentiment_classes=2)

    X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1)

    y_pred, _ = get_logistic_regression_sentiment_prediction(X_test, preprocess=False)
    plot(y_test, y_pred, 'Logistic Regression')

    y_pred, _ = get_decision_tree_sentiment_prediction(X_test, preprocess=False)
    plot(y_test, y_pred, 'Decision Tree')

    y_pred, _ = get_random_forest_sentiment_prediction(X_test, preprocess=False)
    plot(y_test, y_pred, 'Random Forest')




