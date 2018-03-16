import joblib
import numpy as np

from sklearn.linear_model import LogisticRegressionCV

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import prepare_yelp_reviews_dataset_sklearn
from staging.utils.sklearn_utils import create_train_test_set, compute_metrics, make_f1_scorer
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES


if __name__ == '__main__':
    # global constants
    CROSS_VALIDATION = 3

    # classifier hyperparameters
    Cs = 10000 # [10., 10., 1000.]

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
Train set class distribution : [0.32579221 0.67420779]
Test set class distribution : [0.32576698 0.67423302]
[Parallel(n_jobs=2)]: Done   3 out of   3 | elapsed:  3.7min remaining:    0.0s
[Parallel(n_jobs=2)]: Done   3 out of   3 | elapsed:  3.7min finished
Accuracy : 64.76325589643554

************************* Classification Report *************************
             precision    recall  f1-score   support

   negative     0.4373    0.2847    0.3449      1837
   positive     0.7043    0.8230    0.7590      3802

avg / total     0.6173    0.6476    0.6241      5639


************************* Confusion Matrix *************************
negative   positive   
[[ 523 1314]
 [ 673 3129]]

Best C :  [10000.]
Finished training model !
"""