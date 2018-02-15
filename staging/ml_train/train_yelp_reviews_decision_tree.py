import joblib
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import prepare_yelp_reviews_dataset_sklearn
from staging.utils.sklearn_utils import create_train_test_set, compute_metrics, make_f1_scorer
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES

if __name__ == '__main__':
    # global constants
    CROSS_VALIDATION = 5

    # classifier hyperparameters
    max_depth = 4

    reviews_path = resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')
    data, labels = prepare_yelp_reviews_dataset_sklearn(reviews_path, nb_sentiment_classes=3)

    X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1)

    ''' Grid Search for best param_grid '''
    # param_grid = {
    #     'max_depth': [3, 4, 5, 6],
    # }
    # clf = DecisionTreeClassifier(random_state=1, class_weight='balanced')
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
    clf = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced',
                                 random_state=1)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    compute_metrics(y_test, y_preds, target_names=SENTIMENT_CLASS_NAMES)

    clf_path = 'models/sklearn/sentiment/decision_tree.pkl'
    clf_path = construct_data_path(clf_path)

    joblib.dump(clf, clf_path)
    print("Finished training model !")

"""
Vectorizer loaded from saved state !
TF-IDF transformer loaded from saved state !
Shape of tf-idf transformed datased :  (56383, 1575218)
Train set size: (50744, 1575218)
Test set size: (5639, 1575218)
Train set class distribution : [0.14667744 0.17911477 0.67420779]
Test set class distribution : [0.14665721 0.17910977 0.67423302]
Fitting 5 folds for each of 4 candidates, totalling 20 fits
[Parallel(n_jobs=2)]: Done  20 out of  20 | elapsed:  9.7min remaining:    0.0s
...
[Parallel(n_jobs=2)]: Done  20 out of  20 | elapsed:  9.7min finished

Best Parameters :  {'max_depth': 4}
Accuracy : 42.95087781521546

************************* Classification Report *************************
             precision    recall  f1-score   support

   negative     0.1729    0.6252    0.2708       827
   positive     0.7204    0.5008    0.5908      3802

avg / total     0.6226    0.5230    0.5337      4629


************************* Confusion Matrix *************************
negative	positive
[[ 517  308]
 [1896 1904]]

Finished training model !
"""
