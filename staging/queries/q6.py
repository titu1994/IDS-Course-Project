import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "..")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import logging
logging.basicConfig(level=logging.INFO)

from staging import resolve_data_path, construct_data_path

if __name__ == '__main__':
    # change this path if you changed the df path inside "evaluate_ml_restaurants_sentiment.py"
    # NOTE: Can change `ml` to `dl` to get the results from the unbiased Deep Learning models
    df_path = "results/yelp/ml_sentiment_rating_query_result.csv"
    df_path = resolve_data_path(df_path)

    df = pd.read_csv(df_path, encoding='latin-1', header=0)
    df['CorrectSentiment'] = df['ReviewRating'].apply(lambda x: 0 if x <= 3 else 1)

    print(df.info())

    """
    Query :

    Identify how the sentiments relate to the review rating by plotting average ratings against most frequent overall sentiments.
    Sentiment labels, Average Review Rating
    """
    sns.countplot(x='ReviewRating', hue='SentimentLabels', data=df,
                  palette=sns.xkcd_palette(['pale red', 'denim blue']))
    plt.xlabel('Review Ratings')
    plt.ylabel('Instance Counts')
    plt.show()

    g = sns.FacetGrid(df, col='ReviewRating', row='SentimentLabels', hue='CorrectSentiment',
                      palette=sns.xkcd_palette(['pale red', 'denim blue']))
    g.map(sns.countplot, 'ReviewRating', order=None)
    plt.tight_layout()
    plt.show()