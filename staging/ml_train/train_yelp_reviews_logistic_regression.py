import joblib
import numpy as np

from sklearn.linear_model import LogisticRegressionCV

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import prepare_yelp_reviews_dataset_sklearn
from staging.utils.sklearn_utils import create_train_test_set, compute_metrics, make_f1_scorer
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES


if __name__ == '__main__':
    # global constants
    CROSS_VALIDATION = None

    # classifier hyperparameters
    Cs = [10., 10., 1000.]

    reviews_path = resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')
    data, labels = prepare_yelp_reviews_dataset_sklearn(reviews_path, nb_sentiment_classes=3)

    X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1)

    clf = LogisticRegressionCV(Cs, cv=CROSS_VALIDATION, class_weight='balanced',
                               scoring=make_f1_scorer, solver='newton-cg',
                               max_iter=250, n_jobs=2, verbose=10, random_state=1)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    compute_metrics(y_test, y_preds, target_names=SENTIMENT_CLASS_NAMES)
    print('Best C : ', clf.C_)

    clf_path = 'models/sklearn/sentiment/logistic_regression.pkl'
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
[Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:   47.4s
[Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:  1.6min
[Parallel(n_jobs=2)]: Done   7 out of   9 | elapsed:  3.1min remaining:   52.5s
[Parallel(n_jobs=2)]: Done   9 out of   9 | elapsed:  3.8min remaining:    0.0s
[Parallel(n_jobs=2)]: Done   9 out of   9 | elapsed:  3.8min finished
Accuracy : 61.60666784890938

************************* Classification Report *************************
             precision    recall  f1-score   support

   negative     0.2500    0.1524    0.1893       827
   positive     0.7010    0.8514    0.7689      3802

avg / total     0.6204    0.7265    0.6653      4629


************************* Confusion Matrix *************************
negative   positive   
[[ 126  601]
 [ 259 3237]]

Best C :  [  10.   10. 1000.]
Finished training model !
"""