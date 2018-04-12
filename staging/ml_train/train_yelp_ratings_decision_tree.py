import joblib
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import prepare_yelp_ratings_dataset_sklearn
from staging.utils.sklearn_utils import create_train_test_set, compute_metrics, make_f1_scorer
from staging.utils.sklearn_utils import RATINGS_CLASS_NAMES

if __name__ == '__main__':
    # global constants
    CROSS_VALIDATION = 5

    # classifier hyperparameters
    max_depth = 6

    reviews_path = resolve_data_path('datasets/yelp-reviews/reviews.csv')
    data, labels = prepare_yelp_ratings_dataset_sklearn(reviews_path)

    X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1)

    ''' Grid Search for best param_grid '''
    # param_grid = {
    #     'max_depth': [3, 4, 5, 6],
    #     'criterion': ['gini', 'entropy'],
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
    # compute_metrics(y_test, y_preds, target_names=RATINGS_CLASS_NAMES)

    ''' Train final model after hyper-parameter search '''
    clf = DecisionTreeClassifier(max_depth=max_depth,
                                 criterion='gini',
                                 class_weight='balanced',
                                 random_state=1)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    compute_metrics(y_test, y_preds, target_names=RATINGS_CLASS_NAMES)

    clf_path = 'models/sklearn/ratings/decision_tree.pkl'
    clf_path = construct_data_path(clf_path)

    joblib.dump(clf, clf_path)
    print("Finished training model !")

"""
Vectorizer loaded from saved state !
TF-IDF transformer loaded from saved state !
Shape of tf-idf transformed datased :  (56383, 1575218)
Train set class distribution : [0.07110908 0.09936973 0.18959032 0.35346142 0.28646945]
Test set class distribution : [0.07087334 0.09922268 0.18975766 0.35345222 0.2866941 ]
Accuracy : 27.572016460905353

************************* Classification Report *************************
             precision    recall  f1-score   support

          1     0.1288    0.6903    0.2170       155
          2     0.1750    0.0323    0.0545       217
          3     0.2649    0.3205    0.2901       415
          4     0.4555    0.1125    0.1805       773
          5     0.4318    0.4290    0.4304       627

avg / total     0.3615    0.2757    0.2630      2187


************************* Confusion Matrix *************************
1   2   3   4   5   
[[107   1  26   5  16]
 [125   7  46  12  27]
 [179   9 133  30  64]
 [212  17 210  87 247]
 [208   6  87  57 269]]

Finished training model !


"""
