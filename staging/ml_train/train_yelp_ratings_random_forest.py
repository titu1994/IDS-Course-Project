import joblib
import numpy as np

import sys
sys.path.insert(0, "..")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import prepare_yelp_ratings_dataset_sklearn
from staging.utils.sklearn_utils import create_train_test_set, compute_metrics, make_f1_scorer
from staging.utils.sklearn_utils import RATINGS_CLASS_NAMES

if __name__ == '__main__':
    # global constants
    CROSS_VALIDATION = 5

    # classifier hyperparameters
    n_estimators = 500
    max_depth = 6

    reviews_path = resolve_data_path('datasets/yelp-reviews/reviews.csv')
    data, labels = prepare_yelp_ratings_dataset_sklearn(reviews_path)

    X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1)

    ''' Grid Search for best param_grid '''
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [3, 4, 5, 6],
    # }
    # clf = RandomForestClassifier(random_state=1, class_weight='balanced')
    # gclf = GridSearchCV(clf, param_grid, scoring=make_f1_scorer, n_jobs=2,
    #                     cv=CROSS_VALIDATION, verbose=10)
    #
    # gclf.fit(X_train, y_train)
    # print("Best Parameters : ", gclf.best_params_)
    #
    # clf = gclf.best_estimator_
    # y_preds = clf.predict(X_test)
    # compute_metrics(y_test, y_preds, target_names=RATINGS_CLASS_NAMES)

    ''' Train final model after hyper-parameter search '''
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced',
                                 random_state=1)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    compute_metrics(y_test, y_preds, target_names=RATINGS_CLASS_NAMES)

    clf_path = 'models/sklearn/ratings/random_forest.pkl'
    clf_path = construct_data_path(clf_path)

    joblib.dump(clf, clf_path)
    print("Finished training model !")

"""
Vectorizer loaded from saved state !
TF-IDF transformer loaded from saved state !
Shape of tf-idf transformed datased :  (56383, 1575218)
Train set class distribution : [0.07110908 0.09936973 0.18959032 0.35346142 0.28646945]
Test set class distribution : [0.07087334 0.09922268 0.18975766 0.35345222 0.2866941 ]
Accuracy : 46.73068129858253

************************* Classification Report *************************
             precision    recall  f1-score   support

          1     0.3214    0.5806    0.4138       155
          2     0.3709    0.3641    0.3674       217
          3     0.3939    0.4024    0.3981       415
          4     0.5841    0.3325    0.4237       773
          5     0.5169    0.6842    0.5889       627

avg / total     0.4890    0.4673    0.4599      2187


************************* Confusion Matrix *************************
1   2   3   4   5   
[[ 90  30  13   7  15]
 [ 54  79  48   8  28]
 [ 49  58 167  68  73]
 [ 36  31 164 257 285]
 [ 51  15  32 100 429]]

Finished training model !

"""
