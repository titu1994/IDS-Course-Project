import numpy as np
from staging.utils import sklearn_utils

def test_create_train_test_set():
    X = np.zeros((10, 5))
    y = np.zeros((10,))

    X_train, y_train, X_test, y_test = sklearn_utils.create_train_test_set(X, y, test_size=0.1)

    assert X_train.shape == (9, 5)
    assert y_train.shape == (9,)
    assert X_test.shape == (1, 5)
    assert y_test.shape == (1,)

def test_compute_metrics():
    y_true = np.zeros((5,))
    y_true[3:] = 1

    y_pred = np.zeros((5,))
    y_pred[2:] = 1

    acc = sklearn_utils.compute_metrics(y_true, y_pred)

    assert acc == 0.8

def test_compute_class_weight():
    y_true = np.zeros((10,))
    y_true[2:7] = 1

    class_weight = sklearn_utils.compute_class_weight(y_true)
    assert np.array_equal(class_weight, [1.0, 1.0])

    y_true[8] = 1
    class_weight = sklearn_utils.compute_class_weight(y_true)

    assert np.allclose(class_weight, np.array([1.25, 0.8333]), rtol=1e-3)
