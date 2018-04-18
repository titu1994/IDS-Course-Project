import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "..")

import logging
logging.basicConfig(level=logging.INFO)

from staging import resolve_data_path

if __name__ == '__main__':
    # change this path if you changed the df path inside "evaluate_ml_restaurants_sentiment.py"
    # NOTE: Can change `ml` to `dl` below
    df_path = "results/yelp/ml_sentiment_rating_query_result.csv"
    df_path = resolve_data_path(df_path)

    df = pd.read_csv(df_path, encoding='latin-1', header=0)
    df['RestaurantName'] = df['RestaurantName'].apply(lambda x: str(x).lower())

    all_restaurant_names = df['RestaurantName'].unique()

    for name in all_restaurant_names:
        print(name)

    print()
    query = str(input("Input the restaurant name : "))
    print()

    print("Restaurant name:", query)
    print()

    query = query.lower()

    if query not in all_restaurant_names:
        print("This restaurant name '{}' is not present in the database. Please try again.".format(query), file=sys.stderr)
        exit()

    restaurant_df = df[df['RestaurantName'] == query]

    print(restaurant_df)