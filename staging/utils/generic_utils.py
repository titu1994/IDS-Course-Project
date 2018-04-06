import os
import numpy as np


def resolve_data_path(path):
    path1 = "../data/" + path
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path1[1:]):
        return path1[1:]

    path2 = "staging/data/" + path
    if os.path.exists(path2):
        return path2
    else:
        print("File not found ! Seached %s and %s" % (path1, path2))
        return None


def construct_data_path(filename):
    if 'staging' in os.getcwd():
        path = '../data/' + filename

        if os.path.exists(path[1:]):
            path = path[1:]
    else:
        path = '/staging/data/' + filename

    directory = os.path.split(path)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    return path


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def _get_predictions(model, X):
    '''
    Wrapper function to get predictions transparently from either
    SKLearn, XGBoost, Ensemble or Keras models.

    Args:
        model: Model for prediction
        X: input data in correct format for prediction

    Returns:
        predictions
    '''
    if hasattr(model, 'predict_proba'):  # Normal SKLearn classifiers
        pred = model.predict_proba(X)
    elif hasattr(model, '_predict_proba_lr'):  # SVMs
        pred = model._predict_proba_lr(X)
    else:
        pred = model.predict(X)

    if len(pred.shape) == 1:  # for 1-d ouputs
        pred = pred[:, None]

    return pred
