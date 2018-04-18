import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import sys
sys.path.insert(0, "..")

import logging
logging.basicConfig(level=logging.INFO)

from staging import resolve_data_path

if __name__ == '__main__':
    # change this path if you changed the df path inside "evaluate_ml_inspection_review_ratings.py"
    df_path = 'results/yelp/inspection_ratings.csv'
    df_path = resolve_data_path(df_path)

    df = pd.read_csv(df_path, encoding='latin-1', header=0)
    print(df.info())

    pass_df = df[df['#Pass'] != 0]
    conditional_df = df[df['#Conditional'] != 0]
    fail_df = df[df['#Fail'] != 0]

    pass_x = pass_df['#Pass'].values
    pass_y = pass_df['AverageYelpRating'].values
    pass_x, pass_y = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(pass_x, pass_y) if xVal == a])) for xVal in set(pass_x)))

    conditional_x = conditional_df['#Conditional'].values
    conditional_y = conditional_df['AverageYelpRating'].values
    conditional_x, conditional_y = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(conditional_x, conditional_y) if xVal == a])) for xVal in set(conditional_x)))

    fail_x = fail_df['#Fail'].values
    fail_y = fail_df['AverageYelpRating'].values
    fail_x, fail_y = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(fail_x, fail_y) if xVal == a])) for xVal in set(fail_x)))

    fig, ax = plt.subplots(3, 1)

    ax[0].scatter(pass_df['#Pass'], pass_df['AverageYelpRating'], marker='*', label='Pass', c='b')
    ax[0].plot(pass_x, pass_y, c='b', label='Mean', alpha=0.3)
    ax[0].set_xlabel('# Pass')
    ax[0].set_ylabel('Average Yelp Rating')
    ax[0].legend()


    ax[1].scatter(conditional_df['#Conditional'], conditional_df['AverageYelpRating'], label='Conditional', marker='+', c='g')
    ax[1].plot(conditional_x, conditional_y, c='g', label='Mean', alpha=0.3)
    ax[1].set_xlabel('# Conditional')
    ax[1].set_ylabel('Average Yelp Rating')
    ax[1].legend()


    ax[2].scatter(fail_df['#Fail'], fail_df['AverageYelpRating'], label='Fail', marker='.', c='r')
    ax[2].plot(fail_x, fail_y, c='r', label='Mean', alpha=0.3)
    ax[2].set_xlabel('# Fail')
    ax[2].set_ylabel('Average Yelp Rating')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

