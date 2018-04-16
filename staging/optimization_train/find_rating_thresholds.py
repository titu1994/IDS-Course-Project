import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")

from staging import construct_data_path
from staging.utils.optimization_utils import load_optimization_data

(reviews, labels, ratings, restaurant_ids, polarity) = load_optimization_data()

def predict(thresholds, X, y):
    mask_1 = X <= thresholds[0]
    mask_2 = (X > thresholds[0]) & (X <= thresholds[1])
    mask_3 = (X > thresholds[1]) & (X <= thresholds[2])
    mask_4 = (X > thresholds[2]) & (X <= thresholds[3])
    mask_5 = X > thresholds[3]


    preds = np.zeros_like(y)
    preds[mask_1] = 0
    preds[mask_2] = 1
    preds[mask_3] = 2
    preds[mask_4] = 3
    preds[mask_5] = 4

    preds = preds.astype(int)

    return np.mean(np.equal(preds, y))

if __name__ == '__main__':

    best_acc = -1
    best_thresholds = -100

    accuracie_list = []
    threshold_list = []


    """ Full Grid Search over 160,000 combinations (will take an enormous amound of time) """
    # for threshold_1 in np.linspace(-1., -0.6, 20):
    #     for threshold_2 in np.linspace(-0.6, -0.2, 20):
    #         for threshold_3 in np.linspace(-0.2, 0.2, 20):
    #             for threshold_4 in np.linspace(0.2, 0.6, 20):
    #
    #                 thresholds = [threshold_1, threshold_2, threshold_3, threshold_4]
    #                 print("Thresholds : ", thresholds)
    #
    #                 acc = predict(thresholds, polarity, ratings)
    #
    #                 if acc > best_acc:
    #                     best_acc = acc
    #                     best_thresholds = thresholds
    #
    #                 accuracie_list.append(acc)
    #                 threshold_list.append(thresholds)

    """ Take a million samples, and compute accuracy for them all """
    num_tests = int(1e4)
    for i in range(num_tests):
        thresholds = np.random.uniform(-1.0, 1.0, size=4)
        thresholds = sorted(thresholds)

        print("Test # : %d | Thresholds : " % (i + 1), thresholds)

        acc = predict(thresholds, polarity, ratings)

        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds

        accuracie_list.append(acc)
        threshold_list.append(thresholds)

    """
    Copy paste this threshold to the predict function in utils
    """
    print()
    print("Best Accuracy : ", best_acc)
    print("Best Thresholds : ", best_thresholds)

    path = "models/optimization/rating_threshold.txt"
    path = construct_data_path(path)

    # Save the threshold with best accuracy
    with open(path, 'w') as f:
        for threshold in best_thresholds:
            f.write(str(threshold) + "\n")

    x = list(range(len(accuracie_list)))
    plt.scatter(x, accuracie_list)
    plt.xlabel('test run #')
    plt.ylabel('accuracy')
    plt.show()

"""

Best Accuracy :  0.719134531814647
Best Threshold :  0.10092997929979297

"""