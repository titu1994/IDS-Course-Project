import numpy as np
from typing import List

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer, f1_score


SENTIMENT_CLASS_NAMES = ['negative', 'positive']


def create_train_test_set(X: np.ndarray, y: np.ndarray, test_size: float = 0.1) -> (np.ndarray, np.ndarray,
                                                                                    np.ndarray, np.ndarray):
    '''
    Creates a single train test split, where the test size is specified.
    Due to use of stratified K-fold, the class balance is preserved in the split exactly.

    Args:
        X: Input X dataset
        y: Input labels Y
        test_size: Percentage of the dataset to be considered for the split

    Returns:
        a tuple of shape (X_train, y_train, X_test, y_test)
    '''
    X = X.astype('float32')
    y = y.astype('float32')

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
    train_indices, test_indices = next(sss.split(X, y))

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print("Train set size:", X_train.shape)
    print("Test set size:", X_test.shape)

    _, train_distribution = np.unique(y_train, return_counts=True)
    print("Train set class distribution :", train_distribution / float(sum(train_distribution)))

    _, test_distribution = np.unique(y_test, return_counts=True)
    print("Test set class distribution :", test_distribution / float(sum(test_distribution)))

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
