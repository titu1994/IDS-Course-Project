import joblib
import numpy as np

import sys
sys.path.insert(0, "..")

from sklearn.linear_model import LogisticRegressionCV

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import prepare_yelp_ratings_dataset_sklearn
from staging.utils.sklearn_utils import create_train_test_set, compute_metrics, make_f1_scorer
from staging.utils.sklearn_utils import RATINGS_CLASS_NAMES


if __name__ == '__main__':
    # global constants
    CROSS_VALIDATION = 10

    # classifier hyperparameters
    Cs = 10

    reviews_path = resolve_data_path('datasets/yelp-reviews/reviews.csv')
    data, labels = prepare_yelp_ratings_dataset_sklearn(reviews_path)

    X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1)

    clf = LogisticRegressionCV(Cs, cv=CROSS_VALIDATION, class_weight='balanced',
                               scoring=make_f1_scorer, solver='newton-cg',
                               max_iter=250, n_jobs=2, verbose=10, random_state=1)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    compute_metrics(y_test, y_preds, target_names=RATINGS_CLASS_NAMES)
    print('Best C : ', clf.C_)
    print('Best Cs : ', clf.Cs_)

    clf_path = 'models/sklearn/ratings/logistic_regression.pkl'
    clf_path = construct_data_path(clf_path)

    joblib.dump(clf, clf_path)
    print("Finished training model !")


"""
Vectorizer loaded from saved state !
TF-IDF transformer loaded from saved state !
Train set size: (19674, 1575218)
Test set size: (2187, 1575218)
Train set class distribution : [0.07110908 0.09936973 0.18959032 0.35346142 0.28646945]
Test set class distribution : [0.07087334 0.09922268 0.18975766 0.35345222 0.2866941 ]
[Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:   46.8s
[Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:  1.5min
[Parallel(n_jobs=2)]: Done   9 tasks      | elapsed:  3.7min
[Parallel(n_jobs=2)]: Done  14 tasks      | elapsed:  5.5min
[Parallel(n_jobs=2)]: Done  21 tasks      | elapsed:  8.7min
[Parallel(n_jobs=2)]: Done  28 tasks      | elapsed: 11.5min
[Parallel(n_jobs=2)]: Done  37 tasks      | elapsed: 16.6min
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed: 20.7min
[Parallel(n_jobs=2)]: Done  50 out of  50 | elapsed: 22.6min finished
Accuracy : 52.94924554183813

************************* Classification Report *************************
             precision    recall  f1-score   support

          1     0.5887    0.5355    0.5608       155
          2     0.4894    0.3180    0.3855       217
          3     0.4709    0.4289    0.4489       415
          4     0.4989    0.6003    0.5449       773
          5     0.6097    0.5805    0.5948       627

avg / total     0.5308    0.5295    0.5263      2187


************************* Confusion Matrix *************************
1   2   3   4   5   
[[ 83  18  23  20  11]
 [ 29  69  54  47  18]
 [ 14  38 178 165  20]
 [  8  13 104 464 184]
 [  7   3  19 234 364]]

Best C :  [10000. 10000. 10000. 10000. 10000.]
Best Cs :  [1.00000000e-04 7.74263683e-04 5.99484250e-03 4.64158883e-02
 3.59381366e-01 2.78255940e+00 2.15443469e+01 1.66810054e+02
 1.29154967e+03 1.00000000e+04]
Finished training model !
"""