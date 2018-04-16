import joblib
import numpy as np

import sys
sys.path.insert(0, "..")

from sklearn.linear_model import LogisticRegressionCV

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import prepare_yelp_reviews_dataset_sklearn
from staging.utils.sklearn_utils import create_train_test_set, compute_metrics, make_f1_scorer
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES


if __name__ == '__main__':
    # global constants
    CROSS_VALIDATION = 10

    # classifier hyperparameters
    Cs = 10

    reviews_path = resolve_data_path('datasets/yelp-reviews/reviews.csv')
    data, labels = prepare_yelp_reviews_dataset_sklearn(reviews_path, nb_sentiment_classes=2)

    X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1)

    clf = LogisticRegressionCV(Cs, cv=CROSS_VALIDATION, class_weight='balanced',
                               scoring=make_f1_scorer, solver='newton-cg',
                               max_iter=250, n_jobs=2, verbose=10, random_state=1)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    compute_metrics(y_test, y_preds, target_names=SENTIMENT_CLASS_NAMES)
    print('Best C : ', clf.C_)
    print('Best Cs : ', clf.Cs_)

    clf_path = 'models/sklearn/sentiment/logistic_regression.pkl'
    clf_path = construct_data_path(clf_path)

    joblib.dump(clf, clf_path)
    print("Finished training model !")


"""
Vectorizer loaded from saved state !
TF-IDF transformer loaded from saved state !
Train set size: (19674, 1575218)
Test set size: (2187, 1575218)
Train set class distribution : [0.36006913 0.63993087]
Test set class distribution : [0.35985368 0.64014632]
[Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:   54.2s
[Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:  1.7min
[Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:  4.3min finished
Accuracy : 80.93278463648834

************************* Classification Report *************************
             precision    recall  f1-score   support

   negative     0.7673    0.6747    0.7181       787
   positive     0.8288    0.8850    0.8560      1400

avg / total     0.8067    0.8093    0.8063      2187


************************* Confusion Matrix *************************
negative   positive   
[[ 531  256]
 [ 161 1239]]

Best C :  [10000.]
Best Cs :  [1.00000000e-04 7.74263683e-04 5.99484250e-03 4.64158883e-02
 3.59381366e-01 2.78255940e+00 2.15443469e+01 1.66810054e+02
 1.29154967e+03 1.00000000e+04]
Finished training model !
"""