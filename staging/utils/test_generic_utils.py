import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from staging.utils import generic_utils

def test_resolve_path():
    path = "models/README.md"
    resolved_path = generic_utils.resolve_data_path(path)
    assert resolved_path is not None

    path = "models/random.abc"
    with pytest.raises(ValueError):
        resolved_path = generic_utils.resolve_data_path(path)
        assert resolved_path is None

def test_construct_path():
    path = "models/README.md"
    resolved_path = generic_utils.construct_data_path(path)

    assert resolved_path is not None

def test_to_categorical():
    data = np.zeros((5, 1))
    data[1, 0] = 1
    data[3, 0] = 2

    categorical_data = generic_utils.to_categorical(data)
    categorical_data_2 = generic_utils.to_categorical(data, num_classes=3)

    assert categorical_data.shape == (5, 3)
    assert np.array_equal(categorical_data, categorical_data_2)
    assert categorical_data[1, 1] == 1
    assert categorical_data[3, 2] == 1

def test_get_predictions():
    X = np.zeros((10, 2))
    X[5:, :] = 1
    y = np.zeros((10,))
    y[5:] = 1

    model = LogisticRegression(random_state=0)
    model.fit(X, y)

    preds = generic_utils._get_predictions(model, X)
    classes = np.argmax(preds, axis=-1).astype(int)

    assert preds.shape == (10, 2)
    assert np.allclose(classes, y)

    model = LinearSVC(random_state=0)
    model.fit(X, y)

    preds = generic_utils._get_predictions(model, X)
    classes = np.argmax(preds, axis=-1).astype(int)

    assert preds.shape == (10, 2)
    assert np.allclose(classes, y)
