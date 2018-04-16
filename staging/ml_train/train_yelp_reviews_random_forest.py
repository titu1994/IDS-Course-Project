import joblib
import numpy as np

import sys
sys.path.insert(0, "..")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import prepare_yelp_reviews_dataset_sklearn
from staging.utils.sklearn_utils import create_train_test_set, compute_metrics, make_f1_scorer
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES

if __name__ == '__main__':
    # global constants
    CROSS_VALIDATION = 5

    # classifier hyperparameters
    n_estimators = 500
    max_depth = 6

    reviews_path = resolve_data_path('datasets/yelp-reviews/reviews.csv')
    data, labels = prepare_yelp_reviews_dataset_sklearn(reviews_path, nb_sentiment_classes=2)

    X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1)

    ''' Grid Search for best param_grid '''
    # param_grid = {
    #     'n_estimators': [10, 50, 100, 200],
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
    # compute_metrics(y_test, y_preds, target_names=SENTIMENT_CLASS_NAMES)

    ''' Train final model after hyper-parameter search '''
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced',
                                 random_state=1)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    compute_metrics(y_test, y_preds, target_names=SENTIMENT_CLASS_NAMES)

    clf_path = 'models/sklearn/sentiment/random_forest.pkl'
    clf_path = construct_data_path(clf_path)

    joblib.dump(clf, clf_path)
    print("Finished training model !")

"""
Vectorizer loaded from saved state !
TF-IDF transformer loaded from saved state !
Shape of tf-idf transformed datased :  (56383, 1575218)
Train set size: (50744, 1575218)
Test set size: (5639, 1575218)
Train set class distribution : [0.32579221 0.67420779]
Test set class distribution : [0.32576698 0.67423302]
Accuracy : 62.2982798368505

************************* Classification Report *************************
             precision    recall  f1-score   support

   negative     0.4283    0.4698    0.4481      1837
   positive     0.7312    0.6970    0.7137      3802

avg / total     0.6325    0.6230    0.6272      5639


************************* Confusion Matrix *************************
negative   positive   
[[ 863  974]
 [1152 2650]]

Finished training model !

"""
