import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "..")

import seaborn as sns
sns.set_style('white')

import logging
logging.basicConfig(level=logging.INFO)

from staging import resolve_data_path, construct_data_path

if __name__ == '__main__':
    # change this path to the location of the saved sentiment rating query from prior script
    df_path = "results/yelp/ml_sentiment_rating_query_result.csv"
    df_path = resolve_data_path(df_path)

    def clean_name(data):
        return str(data).lower()

    df = pd.read_csv(df_path, header=0, encoding='latin-1')

    df['RestaurantName'] = df['RestaurantName'].apply(clean_name)

    print(df.info())
    print()

    # change path to dataset of food inspections from city of chicago
    inspection_path = "datasets/yelp-reviews/Food_Inspections.csv"
    inspection_path = resolve_data_path(inspection_path)

    inspection_df = pd.read_csv(inspection_path, header=0, encoding='latin-1')
    inspection_df.dropna(inplace=True)

    inspection_df['DBA Name'] = inspection_df['DBA Name'].apply(clean_name)

    unique_restaurant_names = df['RestaurantName'].unique()

    def check_inspection_for_name(frame):
        name = frame['DBA Name'].lower()

        if name in unique_restaurant_names:
            return 1.
        else:
            return np.nan

    inspection_df['NameExists'] = inspection_df.apply(check_inspection_for_name, axis=1)
    inspection_df.dropna(inplace=True)

    print(inspection_df.info())
    print()


    merged_df = pd.merge(df, inspection_df, left_on='RestaurantName', right_on='DBA Name')  # type: pd.DataFrame
    merged_df.dropna(inplace=True)

    merged_df = merged_df[['RestaurantName', 'SentimentLabels', 'ReviewRating',
                           'Address', 'Results',]]

    def apply_pass(result):
        if result == 'Pass':
            return 1
        else:
            return 0

    def apply_fail(result):
        if result == 'Fail':
            return 1
        else:
            return 0

    def apply_pass_w_conditions(result):
        if result == 'Pass w/ Conditions':
            return 1
        else:
            return 0

    merged_df['Pass'] = merged_df['Results'].apply(apply_pass)
    merged_df['Fail'] = merged_df['Results'].apply(apply_fail)
    merged_df['PassWConditions'] = merged_df['Results'].apply(apply_pass_w_conditions)

    print(merged_df.info())
    print()

    aggregate_df = merged_df.groupby(by=['RestaurantName', 'Address', 'Results'], as_index=False, sort=False)
    aggregate_df = aggregate_df.agg({
        'ReviewRating': ['mean'],
        'Pass': ['sum'],
        'Fail': ['sum'],
        'PassWConditions': ['sum']
    }).reset_index()  # type: pd.DataFrame

    # fix the names
    aggregate_df.columns = [x[0] for x in aggregate_df.columns.ravel()]

    aggregate_df = aggregate_df[['RestaurantName', 'Address', 'ReviewRating', 'Pass',
                                 'PassWConditions', 'Fail']]

    aggregate_df = aggregate_df.rename(columns={
        'ReviewRating': 'AverageYelpRating',
        'Pass': '#Pass',
        'Fail': '#Fail',
        'PassWConditions': '#Conditional',
    })

    print(aggregate_df.info())
    print()

    print(aggregate_df.head(10))

    save_path = 'results/yelp/inspection_ratings.csv'
    save_path = construct_data_path(save_path)

    aggregate_df.to_csv(save_path, encoding='utf-8', index_label='id')

    print("Finished saving files.")