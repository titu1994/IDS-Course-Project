import numpy as np
import os
from sklearn.linear_model import LogisticRegression

from staging.utils import sklearn_utils

def test_create_train_test_set():
    X = np.zeros((100, 2))
    y = np.zeros((100,))
    X[50:, :] = 1
    y[50:] = 1

    X_train, y_train, X_test, y_test = sklearn_utils.create_train_test_set(X, y, test_size=0.5)

    assert X_train.shape == (50, 2)
    assert y_train.shape == (50,)
    assert X_test.shape == (50, 2)
    assert y_test.shape == (50,)

    X_train, y_train, X_test, y_test = sklearn_utils.create_train_test_set(X, y, test_size=0.5,
                                                                           rebalance_class_distribution=True,
                                                                           cache=True)

    assert X_train.shape == (50, 2)
    assert y_train.shape == (50,)
    assert X_test.shape == (50, 2)
    assert y_test.shape == (50,)


def test_compute_metrics():
    y_true = np.zeros((5,))
    y_true[3:] = 1

    y_pred = np.zeros((5,))
    y_pred[2:] = 1

    acc = sklearn_utils.compute_metrics(y_true, y_pred, target_names=['negative', 'positive'])

    assert acc == 0.8

def test_compute_class_weight():
    y_true = np.zeros((10,))
    y_true[2:7] = 1

    class_weight = sklearn_utils.compute_class_weight(y_true)
    assert np.array_equal(class_weight, [1.0, 1.0])

    y_true[8] = 1
    class_weight = sklearn_utils.compute_class_weight(y_true)

    assert np.allclose(class_weight, np.array([1.25, 0.8333]), rtol=1e-3)


def test_make_f1_scorer():
    X = np.zeros((10, 2))
    y = np.zeros((10,))
    X[5:, :] = 1
    y[5:] = 1

    model = LogisticRegression(random_state=0)
    model.fit(X, y)

    f1_score = sklearn_utils.make_f1_scorer(model, X, y)

    assert f1_score == 1.0
