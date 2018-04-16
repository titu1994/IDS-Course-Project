import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")

from staging import construct_data_path
from staging.utils.optimization_utils import load_optimization_data

(reviews, labels, ratings, restaurant_ids, polarity) = load_optimization_data()

def predict(threshold, X, y):
    preds = (X > threshold).astype(int)
    return np.mean(np.equal(preds, y))

if __name__ == '__main__':

    best_acc = -1
    best_threshold = -100

    accuracies = []
    thresholds = []

    for threshold in np.linspace(-0.999, 0.999, 100000):
        print("Trying threshold = {:.3f}".format(threshold))

        acc = predict(threshold, polarity, labels)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

        accuracies.append(acc)
        thresholds.append(threshold)


    """
    Copy paste this threshold to the predict function in utils
    """
    print()
    print("Best Accuracy : ", best_acc)
    print("Best Threshold : ", best_threshold)

    path = "models/optimization/sentiment_threshold.txt"
    path = construct_data_path(path)

    # Save the threshold with best accuracy
    with open(path, 'w') as f:
        f.write(str(best_threshold))


    plt.plot(thresholds, accuracies)
    plt.xlabel('threshold')
    plt.ylabel('accuracy')
    plt.show()

"""

Best Accuracy :  0.719134531814647
Best Threshold :  0.10092997929979297

"""