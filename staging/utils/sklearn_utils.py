import os
import numpy as np
from typing import List
from hashlib import sha1

from scipy.sparse import issparse, load_npz, save_npz
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight as _compute_class_weight

from staging import to_categorical, _get_predictions, construct_data_path

SENTIMENT_CLASS_NAMES = ['negative', 'positive']
SENTIMENT_CLASS_PRIORS = [0.14665721, 0.17910977, 0.67423302]


def create_train_test_set(X: np.ndarray, y: np.ndarray,
                          test_size: float = 0.1,
                          rebalance_class_distribution: bool = False,
                          cache: bool = False) -> (np.ndarray,
                                                   np.ndarray,
                                                   np.ndarray,
                                                   np.ndarray):
    '''
    Creates a single train test split, where the test size is specified.
    Due to use of stratified K-fold, the class balance is preserved in the split exactly.

    Args:
        X: Input X dataset
        y: Input labels Y
        test_size: Percentage of the dataset to be considered for the split
        rebalance_class_distribution: Use oversampling techniques to balance classes

    Returns:
        a tuple of shape (X_train, y_train, X_test, y_test)
    '''
    X = X.astype('float32')
    y = y.astype('float32')

    if cache:
        if not issparse(X):
            hash_val = sha1(X.tostring()).hexdigest()
            ext = ".npy"

            save_fn = np.save
            load_fn = np.load
        else:
            hash_val = sha1(X.data).hexdigest()
            ext = ".npz"

            save_fn = save_npz
            load_fn = load_npz

        base_path = 'datasets/cache/sentiment/%s/' % hash_val
        cache_path = construct_data_path(base_path)

        if os.path.exists(cache_path + 'x_train' + ext):
            X_train = load_fn(cache_path + 'x_train' + ext)
            y_train = np.load(cache_path + 'y_train.npy')
            X_test = load_fn(cache_path + 'x_test' + ext)
            y_test = np.load(cache_path + 'y_test.npy')

            print("Train set size:", X_train.shape)
            print("Test set size:", X_test.shape)

            if y_train.ndim > 1 and y_train.shape[-1] > 1:
                y_train_temp = np.argmax(y_train, axis=-1)
            else:
                y_train_temp = y_train

            _, train_distribution = np.unique(y_train_temp, return_counts=True)
            print("Train set class distribution :", train_distribution / float(sum(train_distribution)))

            if y_test.ndim > 1 and y_test.shape[-1] > 1:
                y_test_temp = np.argmax(y_test, axis=-1)
            else:
                y_test_temp = y_test

            _, test_distribution = np.unique(y_test_temp, return_counts=True)
            print("Test set class distribution :", test_distribution / float(sum(test_distribution)))

            print("Datasets loaded from cache !")
            return (X_train, y_train, X_test, y_test)
    else:
        hash_val = None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
    train_indices, test_indices = next(sss.split(X, y))

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print("Train set size:", X_train.shape)
    print("Test set size:", X_test.shape)

    if y_train.ndim > 1 and y_train.shape[-1] > 1:
        y_train_temp = np.argmax(y_train, axis=-1)
    else:
        y_train_temp = y_train

    _, train_distribution = np.unique(y_train_temp, return_counts=True)
    print("Train set class distribution :", train_distribution / float(sum(train_distribution)))

    if rebalance_class_distribution:
        print("Oversampling data to overcome class imbalance")
        smote = SMOTE(random_state=0, kind='borderline2', n_jobs=4)

        if y_train.ndim > 1 and y_train.shape[-1] > 1:
            num_classes = y_train.shape[-1]
            X_train, y_train = smote.fit_sample(X_train, y_train_temp)

            y_train_temp = y_train
            y_train = y_train.reshape((-1, 1))
            y_train = to_categorical(y_train, num_classes=num_classes)
        else:
            X_train, y_train = smote.fit_sample(X_train, y_train_temp)
            y_train_temp = y_train

        _, train_distribution = np.unique(y_train_temp, return_counts=True)
        print("Rebalenced Train set class distribution :", train_distribution / float(sum(train_distribution)))

    if y_test.ndim > 1 and y_test.shape[-1] > 1:
        y_test_temp = np.argmax(y_test, axis=-1)
    else:
        y_test_temp = y_test

    _, test_distribution = np.unique(y_test_temp, return_counts=True)
    print("Test set class distribution :", test_distribution / float(sum(test_distribution)))

    if cache:
        print("Saving dataset to cache")
        base_path = 'datasets/cache/sentiment/%s/' % hash_val
        cache_path = construct_data_path(base_path)

        save_fn(cache_path + 'x_train' + ext, X_train)
        np.save(cache_path + 'y_train.npy', y_train)
        save_fn(cache_path + 'x_test' + ext, X_test)
        np.save(cache_path + 'y_test.npy', y_test)

    return (X_train, y_train, X_test, y_test)


def compute_metrics(y_true: np.ndarray, y_preds: np.ndarray, target_names: List[str] = None):
    '''
    Computes and prints a report of the performance of this model

    Args:
        y_true: the true labels
        y_preds: the predicted labels
        target_names: names of the labels for the report
    '''
    acc = accuracy_score(y_true, y_preds)
    print('Accuracy :', acc * 100)
    print()

    print('*' * 25, 'Classification Report', '*' * 25)
    report = classification_report(y_true, y_preds, labels=[0, 2], target_names=target_names, digits=4)
    print(report)
    print()

    print('*' * 25, 'Confusion Matrix', '*' * 25)
    if target_names is not None:
        strng = ''
        for name in target_names:
            strng = strng + '%s   ' % name
        print(strng)

    conf = confusion_matrix(y_true, y_preds, labels=[0, 2])
    print(conf)
    print()


def make_f1_scorer(clf, X, y):
    '''
    Create an f1 scoring function
    Returns:
        a scoring function
    '''
    y = y.astype('uint32')
    y_pred = clf.predict(X)
    return f1_score(y, y_pred, average='micro')


def compute_class_weight(y_true: np.ndarray) -> np.ndarray:
    '''
    Simple wrapper to compute the class weights of the dataset

    Args:
        y_true: the full dataset labels

    Returns:
        class_weight_vect : ndarray, shape (n_classes,)
            Array with class_weight_vect[i] the weight for i-th class
    '''
    return _compute_class_weight('balanced', np.unique(y_true), y_true)
