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
from staging.utils.keras_utils import prepare_yelp_reviews_dataset_keras
from staging.utils.keras_utils import MAX_SEQUENCE_LENGTH, MAX_NB_WORDS
from staging.utils.sklearn_utils import create_train_test_set, compute_metrics, make_f1_scorer
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES

from staging.dl_eval.predict_dl_sentiment import get_lstm_sentiment_prediction
from staging.dl_eval.predict_dl_sentiment import get_multiplicative_lstm_sentiment_prediction
from staging.dl_eval.predict_dl_sentiment import get_malstm_fcn_sentiment_prediction


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
    data, labels, _ = prepare_yelp_reviews_dataset_keras(reviews_path, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH,
                                                      nb_sentiment_classes=2)

    X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1)
    y_test = np.argmax(y_test, axis=-1)

    y_pred, _ = get_lstm_sentiment_prediction(X_test, preprocess=False)
    plot(y_test, y_pred, 'LSTM RNN')

    y_pred, _ = get_multiplicative_lstm_sentiment_prediction(X_test, preprocess=False)
    plot(y_test, y_pred, 'Multiplicative LSTM RNN')

    y_pred, _ = get_malstm_fcn_sentiment_prediction(X_test, preprocess=False)
    plot(y_test, y_pred, 'Multivariate Attention LSTM FCN')




